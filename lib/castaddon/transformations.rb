# This class provides an extension to the CAST node class, which
# is a parent class for all other CAST classes. The extension
# consists of three different types of methods:
# * Methods starting with +transform_+, handling the major code transformations.
# * Methods to obtain information on variables, such as their direction and whether they are defined or not.
# * Helper methods, among others those that indicate whether a node is of a certain class.
class C::Node
	
	# Method to enable the use of local memory for a list of
	# array variables which is given as an argument to the method.
	# The method walks through the node. First, it checks whether: 
	# * The node represents an array access (+node.index?+)
	# * The node has a parent node (+node.parent+)
	# Then, the method loops over all array variables. It checks
	# one more thing: whether the array variable's name is the
	# same as the name found in the array access node.
	#
	# If all conditions are met, the method performs two replacements:
	# * The variable name is changed to correspond to local memory
	# * The index names are changed to correspond to local indices
	#
	# The method performs the transformation on the node itself.
	# Any old data is thus lost.
	def transform_use_local_memory(arrays)
		self.preorder do |node|
			if (node.index?) && (node.parent)
				arrays.each do |array|
					if node.variable_name == array.name
						node.variable_name = Bones::LOCAL_MEMORY+'_'+array.name
						array.species.dimensions.each_with_index do |dimension,num_dimension|
							node.replace_variable(Bones::GLOBAL_ID+'_'+num_dimension.to_s,Bones::LOCAL_ID+'_'+num_dimension.to_s)
						end
					end
				end
			end
		end
	end
	
	# This method transforms multi-dimensional arrays into 1D
	# arrays. The target array variable list is given as argument.
	# The method walks through the node. First, it checks whether:
	# * The node represents an array access (+node.index?+)
	# * The node has a parent node (+node.parent+)
	# Then, the method loops over all array variables. It then
	# checks two more things:
	# * Whether the given name is the same as the name found in the array access node (+node.variable_name == array.name+)
	# * Whether the dimensions of the given array are the same as the dimensions of the node (+node.dimension == array.dimension+)
	#
	# Then, the method is ready to perform the flattening. It
	# first gets the index for the first dimension and then
	# iterates over all remaining dimensions. For those dimensions,
	# the index is multiplied by the size of the previous
	# dimension.
	#
	# The method performs the transformation on the node itself.
	# Any old data is thus lost.
	def transform_flatten(array)
		self.preorder do |node|
			if (node.index?) && (node.parent)
				if (node.variable_name == array.name) && (node.dimension == array.dimensions)
					
					# Compute the new index
					results = array.species.dimensions.each_with_index.map { |d,n| '('+node.index_at_dimension(n).to_s+')'+array.factors[n] }
					replacement = array.name+'['+results.join(' + ')+']'
					
					# Replace the node
					node.replace_with(C::Index.parse(replacement))
				end
			end
		end
	end
	
	# Method to transform array accesses into register accesses.
	# This is only valid for the local loop body and could have
	# been done by the actual compiler in a number of cases.
	def transform_substitution(array,inout)
		replacement = 'register_'+array.name
		original_name = ''
		
		# Change the variable names
		self.stmts.preorder do |node|
			if (node.index?) && (node.parent)
				
				# First time replacement
				if original_name == ''
					if node.variable_name == array.name
						node.replace_with(C::Variable.parse(replacement))
						original_name = node.to_s
					end
					
				# Second, third, etc. replacement
				else
					if original_name == node.to_s
						node.replace_with(C::Variable.parse(replacement))
					end
				end
			end
		end
		
		# Add prologue and epilogue code
		if original_name != ''
			if inout
				self.stmts[0].insert_prev(C::Declaration.parse(array.type_name+' '+replacement+'='+original_name+';'))
			else
				self.stmts[0].insert_prev(C::Declaration.parse(array.type_name+' '+replacement+';'))
			end
			self.stmts[self.stmts.length-1].insert_next(C::ExpressionStatement.parse(original_name+' = '+replacement+';'))
		end
	end
	
	# Method to shuffle a 2D array access (e.g. transform from
	# A[i][j] into A[j][i]).
	def transform_shuffle(arrays)
		arrays.each do |array|
		
			# Change the variable names
			self.stmts.preorder do |node|
				if (node.index?) && (node.parent)
					if node.variable_name == array.name && node.expr.index?
						replacement = node.variable_name.to_s+'['+node.index.to_s+']['+node.expr.index.to_s+']'
						node.replace_with(C::Index.parse(replacement))
					end
				end
			end
		
		end
	end
	
	# Method to merge the computations of multiple threads.
	def transform_merge_threads(amount,excludes)
		self.preorder do |node|
			if node.statement?
				replacement = C::NodeArray.new
				amount.times do |i|
					replacement.push(node.clone.rename_variables('_m'+i.to_s,excludes))
				end
				node.replace_with(replacement)
			end
		end
	end
	def rename_variables(suffix,excludes)
		self.preorder do |node|
			if (node.variable? || node.declarator?) && !(excludes.include?(node.name)) && (!node.parent.call?)
				node.name = node.name+suffix
			end
		end
	end
	
	# This method provides the transformations necessary to
	# perform reduction type of operations. The transformations
	# involved in this function are on variable names and index
	# locations. The argument +id+ specifies which transformation
	# to be performed.
	#
	# Accepted inputs at this point: 2, 3 and 4 (CUDA/OPENCL)
	# Also accepted input: 8 (CUDA), 9 (OPENCL) (to create an atomic version of the code)
	# TODO: Complete the atomic support, e.g. add support for multiplications and ternary operators
	def transform_reduction(input_variable,shared_variable,id)
		
		# Pre-process assign-add type constructions
		self.preorder do |node|
			if node.addassign?
				node.replace_with(C::Assign.parse(node.lval.to_s+'='+node.lval.to_s+'+'+node.rval.to_s))
			end
		end
		
		# Create atomic code
		if id == 8 || id == 9
			function_name = (id == 8) ? 'atomicAdd' : 'atomic_add'
			self.preorder do |node|
				if node.assign?
					if node.lval.index? && node.lval.variable_name == shared_variable.name
						if node.rval.add?
							if node.rval.expr1.variable_name == shared_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr1.to_s+','+node.rval.expr2.to_s+')'))
							elsif node.rval.expr2.variable_name == shared_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr2.to_s+','+node.rval.expr1.to_s+')'))
							end
						elsif node.rval.subtract?
							if node.rval.expr1.variable_name == shared_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr1.to_s+',-'+node.rval.expr2.to_s+')'))
							elsif node.rval.expr2.variable_name == shared_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr2.to_s+',-'+node.rval.expr1.to_s+')'))
							end
						elsif node.assign?
							
						else
							raise_error('Unsupported atomic reduction operator: '+node.rval.type.inspect)
						end
					end
				end
			end
			return self
		else
		
			# Split the statement into an operation, the input, and the output
			results = []
			operation = self.stmts[0].expr.rval.class
			[self.stmts[0].expr.rval.expr1.detach,self.stmts[0].expr.rval.expr2.detach].each do |nodes|
				nodes.preorder do |node|
					if (node.index?)
						results[0] = nodes if node.variable_name == input_variable.name
						results[1] = nodes if node.variable_name == shared_variable.name
					end
				end
			end
			
			# Process the input part
			results[0].preorder do |node|
				if (node.index?) && (node.variable_name == input_variable.name)
					temporary = C::Variable.parse(Bones::VARIABLE_PREFIX+'temporary')
					results[0] = C::Index.parse(Bones::LOCAL_MEMORY+'['+Bones::VARIABLE_PREFIX+'offset_id]') if id == 3
					results[0] = temporary if id == 5
					if id == 2 || id == 4
						if node.parent
							node.replace_with(temporary)
						else
							results[0] = temporary
						end
					end
				end
			end
			
			# Process the output part
			results[1] = C::Variable.parse(Bones::PRIVATE_MEMORY)                          if id == 2 || id == 5
			results[1] = C::Index.parse(Bones::LOCAL_MEMORY+'['+Bones::LOCAL_ID+']')       if id == 3
			results[1] = '0'                                                               if id == 4
			
			# Merge the results together with the operation
			return C::Expression.parse(results[1].to_s+'+'+results[0].to_s) if id == 3 || id == 5
			case operation.to_s
				when 'C::Add'      then return C::Expression.parse(results[1].to_s+'+'+results[0].to_s)
				when 'C::Subtract' then return C::Expression.parse(results[1].to_s+'-'+results[0].to_s)
				else raise_error('Unsupported reduction operation '+operation.to_s+'.')
			end
		end
	end
	
end

