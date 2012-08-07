# This class provides an extension to the CAST node class, which
# is a parent class for all other CAST classes. The extension
# consists of three different types of methods:
# * Methods starting with +transform_+, handling the major code transformations.
# * Methods to obtain information on variables, such as their direction and whether they are defined or not.
# * Helper methods, among others those that indicate whether a node is of a certain class.
class C::Node
	
	# Pre-process method. It currently pre-processes a piece of
	# code (typically the kernel code) to replace particular
	# code structures with others, which can be handled (better)
	# by Bones. For now, the pre-process method performs the
	# following transformations:
	# * Replaces all incrementors (i++) outside for loops with an assignment (i=i+1).
	# * Replaces all decrementors (i--) outside for loops with an assignment (i=i-1).
	def preprocess
		self.preorder do |node|
			if node.postinc? || node.preinc?
				node.replace_with(C::AssignmentExpression.parse(node.expr.to_s+' = '+node.expr.to_s+' + 1')) unless node.parent.for_statement?
			elsif node.postdec? || node.predec?
				node.replace_with(C::AssignmentExpression.parse(node.expr.to_s+' = '+node.expr.to_s+' - 1')) unless node.parent.for_statement?
			end
		end
	end
	
	# Method to obtain a list of all functions in the code. If
	# no functions can be found, an empty array is returned. For
	# every function found, the function itself is pushed to the
	# list. This method also makes sure that there is at least one
	# function with the name 'main'. If this is not the case, an
	# error is raised.
	def get_functions
		includes_main = false
		function_list = []
		self.preorder do |node|
			if node.function_definition?
				function_list.push(node)
				includes_main = true if (node.name == 'main' || node.name == Bones::VARIABLE_PREFIX+'main')
			end
		end
		raise_error('No "main"-function detected in the input file') if !includes_main
		return function_list
	end
	
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
	def transform_reduction(input_variable,output_variable,id)
		
		# Pre-process assign-add type constructions
		if self.stmts[0].expr.addassign?
			self.stmts[0].expr.replace_with(C::Assign.parse(self.stmts[0].expr.lval.to_s+'='+self.stmts[0].expr.lval.to_s+'+'+self.stmts[0].expr.rval.to_s))
		end
		
		# Create atomic code
		if id == 8 || id == 9
			function_name = (id == 8) ? 'atomicAdd' : 'atomic_add'
			self.preorder do |node|
				if node.assign?
					if node.lval.index? && node.lval.variable_name == output_variable.name
						if node.rval.add?
							if node.rval.expr1.variable_name == output_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr1.to_s+','+node.rval.expr2.to_s+')'))
							elsif node.rval.expr2.variable_name == output_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr2.to_s+','+node.rval.expr1.to_s+')'))
							end
						elsif node.rval.subtract?
							if node.rval.expr1.variable_name == output_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr1.to_s+',-'+node.rval.expr2.to_s+')'))
							elsif node.rval.expr2.variable_name == output_variable.name
								node.replace_with(C::Call.parse(function_name+'(&'+node.rval.expr2.to_s+',-'+node.rval.expr1.to_s+')'))
							end
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
						results[1] = nodes if node.variable_name == output_variable.name
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
	
	# This method returns the complexity of a piece of code in
	# terms of the amount of ALU operations (multiplications,
	# additions, etc.).
	def get_complexity
		count = 0
		self.preorder do |node|
			count += 1 if node.alu?
		end
		return count
	end
	
	# This method returns the type of a variable (e.g. int, float).
	# The method requires the name of a variable as an argument.
	# It first tries to find a declaration for the variable in
	# by walking through the node. If it cannot find it, it will
	# search for a parameter definition from a function call. If
	# that cannot be found either, the method will return 'nil',
	# meaning that the variable is not defined at all in the
	# current node.
	def variable_type(variable_name)
		self.preorder do |node|
			if node.declarator? || node.parameter?
				return node.type if node.name == variable_name
			end
		end
		return nil
	end
	
	# This method returns the sizes of a variable as defined
	# at the initialization of the array. There are multiple
	# possibilities:
	# * Static arrays (e.g. int array[12])
	# * Static arrays with defines (e.g. int input[N1][N2][N3])
	# * Variable length arrays (e.g. float temp[n][m])
	# * Dynamically allocated arrays (e.g. int *a = (int *)malloc(size*4))
	def size(variable_name)
		self.preorder do |node|
			if node.declarator? && node.name == variable_name
				if node.indirect_type
					if node.indirect_type.array?
						return node.indirect_type.lengths
					elsif node.indirect_type.pointer?
						node.preorder do |inner_node|
							if inner_node.call? && inner_node.expr.name == 'malloc'
								if !node.indirect_type.type # This is a check to ensure single-pointer only
									string = '('+inner_node.args.to_s+'/sizeof('+node.type.to_s.gsub('*','').strip+'))'
									string.gsub!(/sizeof\(int\)\/sizeof\(int\)/,'1')
									string.gsub!(/sizeof\(unsigned int\)\/sizeof\(unsigned int\)/,'1')
									string.gsub!(/sizeof\(char\)\/sizeof\(char\)/,'1')
									string.gsub!(/sizeof\(unsigned char\)\/sizeof\(unsigned char\)/,'1')
									string.gsub!(/sizeof\(double\)\/sizeof\(double\)/,'1')
									string.gsub!(/sizeof\(float\)\/sizeof\(float\)/,'1')
									return [string]
								end
							end
						end
					end
				end
			end
		end
		return []
	end
	
	# This is a helper method which calls itself recursively,
	# depending on the dimensions of the variable. It stores
	# the resulting array sizes in an array 'result'.
	def lengths(result = [])
		found = '('+self.length.to_s+')'
		result.push(found)
		return (self.type && self.type.array?) ? self.type.lengths(result) : result
	end
	
	# This method returns a list of undefined variables in the
	# node. It walks the node tree until it finds a node that
	# full-fills the following:
	# * The node is a variable (+node.variable?+)
	# * The variable is not in a function call (+!node.parent.call?+)
	# * The variable is not defined in the code (+!self.variable_type(node.name)+)
	def undefined_variables
		variables = []
		self.preorder do |node|
			variables.push(node.name) if (node.variable?) && (!node.parent.call?) && (!self.variable_type(node.name))
		end
		return variables.uniq
	end
	
	# This method finds the direction of a variable based on the
	# node information. The direction of a variable can be either:
	# * +in:+ - The variable is accessed read-only.
	# * +out:+ - The variable is accessed write-only.
	#
	# The method takes the name of a variable and walks through
	# the node to search for expressions (assignments and binary
	# expressions). For each expression it takes the first and
	# second part of the expression and stores it in a list.
	# Afterwards, the expressions in the list are analysed for
	# occurrences of the variable.
	#
	# The method raises an error if the variable does not appear
	# at all: it is neither input nor output.
	def direction(variable_name)
		result = {:in => false, :out => false }
		expressions = {:in => [], :out => []}
		output_nodes = []
		self.preorder do |node|
			
			# First find out if the current node actually contains the target variable somewhere
			name_match = false
			node.preorder do |match_node|
				name_match = true if (match_node.variable?) && (match_node.name == variable_name)
			end
			
			# If there is a match and the current node is of an assignment/binary/declarator type, we can start processing
			if (name_match) && (node.assignment_expression? || node.binary_expression? || node.declarator?)
				
				# First find out if this node can be considered an input (see sum/acc/temp register variable problem - chunk/example1 vs chunk/example5)
				possible_input = true
				node.preorder do |test_node|
					output_nodes.each do |output_node|
						possible_input = false if test_node =~ output_node
					end
				end
				
				# Store the node's data in a list (input/output lists are separated)
				if node.assignment_expression?
					output_nodes << node.lval
					expressions[:out] << node.lval.remove_index
					expressions[:in] << node.rval                if possible_input
					if !node.assign?
						expressions[:in] << node.lval              if possible_input
					end
				elsif node.binary_expression?
					expressions[:in] << node.expr1               if possible_input
					expressions[:in] << node.expr2.remove_index  if possible_input
				elsif node.declarator? && node.init
					expressions[:in] << node.init                if possible_input
				end
			end
		end
		
		# Set the result according to the list of nodes
		expressions.each do |key,expression_list|
			expression_list.each do |expression|
				expression.preorder do |node|
					if (node.variable?) && (node.name == variable_name)
						result[key] = true
					end
				end
			end
		end
		
		# Return the result
		return Bones::INOUT if result[:in] && result[:out]
		return Bones::INPUT if result[:in]
		return Bones::OUTPUT if result[:out]
		raise_error('Variable "'+variable_name+'" is neither input nor output')
	end
	
	# This method walks through the node and finds the first
	# for-loop. If it is found, it returns the contents of the
	# for-loop and the name of the loop variable. Obtaining the
	# loop variable is conditional because it can either be an
	# assignment ('k=0') or a variable definition ('int k=0').
	#
	# The method raises an error when no for-loop can be found.
	# It also raises an error if the loop is not in canonical
	# form.
	def remove_loop(from,to)
		bones_common = Bones::Common.new()
		self.preorder do |node|
			if node.for_statement?
				from_statement = (node.init.assign?) ? node.init.rval : node.init.declarators[0].init
				from_loop = (from_statement.variable?) ? from_statement.name : from_statement.to_s
				to_loop = (node.cond.expr2.variable?) ? node.cond.expr2.name : ((node.cond.expr2.intliteral?) ? node.cond.expr2.val.to_s : node.cond.expr2.to_s)
				to_loop = to_loop.gsub(/\s/,'')
				to_loop = '('+to_loop+')-1' if node.cond.less?
				to_loop = bones_common.simplify(to_loop)
				from_loop = bones_common.simplify(from_loop)
				puts Bones::WARNING+'The loop iterator starts at: "'+from_loop+'" (expected "'+from+'")' if from_loop != from
				puts Bones::WARNING+'The loop iterator ends at: "'+to_loop+'" (expected "'+to+'")' if to_loop != to
				raise_error('The loop increment must be 1') if !(node.iter.unit_increment?)
				name = (node.init.assign?) ? node.init.lval.name : node.init.declarators.first.name
				return node.stmt, name
			end
		end
		raise_error('Unexpected number of for-loops')
	end
	
	# This method searches for a variable name in the node and
	# replaces it with the method's argument, which is given as
	# a string. The node itself is modified. The method checks
	# whether:
	# * The node is a variable (+node.variable?+)
	# * The variable has the correct name (+node.name == variable_name+)
	# * The variable is not in a function call (+!node.parent.call?+)
	def replace_variable(variable_name,replacement)
		self.preorder do |node|
			node.name = replacement if (node.variable?) && (node.name == variable_name) && (!node.parent.call?)
		end
	end
	
	# This method searches for a target node and replaces it with
	# a replacement node. Both the target node and the replacement
	# node are given as arguments to the method. The method walks
	# through the node and checks whether:
	# * The node's class is the same as the target class (+node.class == target.class+)
	# * The node has a parent (+node.parent != nil+)
	# * The node is equal to the target node (+node.match?(target)+)
	# If all checks are successful, the node will be replaced with
	# the replacement node and the method will return immediately.
	#
	# The method returns itself if the target node cannot be
	# found.
	def seach_and_replace_node(target,replacements)
		self.preorder do |node|
			if (node.class == target.class) && (node.parent != nil) && (node.match?(target))
				node.replace_with(replacements)
				return self
			end
		end
		return self
	end
	
	# This method searches for a target node and checks whether it
	# exists. The input to the method is the target node. The method
	# walks through the node and checks whether:
	# * The node's class is the same as the target class (+node.class == target.class+)
	# * The node has a parent (+node.parent != nil+)
	# * The node is equal to the target node (+node.match?(target)+)
	# If all checks are successful, the method will return the value
	# 'true' immediately. If the target node cannot be found, the
	# method returns 'false'.
	def node_exists?(target)
		self.preorder do |node|
			if (node.class == target.class) && (node.parent != nil) && (node.match?(target))
				return true
			end
		end
		return false
	end
	
	# This method searches for a target function call and replaces
	# it with another. Both the target and the replacement function
	# call are given as arguments to the method. The method walks
	# through the node and checks whether:
	# * The node's class is the same as the target class (+node.class == target.class+)
	# * The node has a parent which is a function call (+node.parent.call?+)
	# * The node is equal to the target node (+node.match?(target)+)
	# If all checks are successful, the node will be replaced with
	# the replacement node. The method will continue searching for
	# other occurrences of the function call.
	#
	# The method returns itself.
	def seach_and_replace_function_call(target,replacements)
		self.preorder do |node|
			if (node.class == target.class) && (node.parent.call?) && (node.match?(target))
				node.replace_with(replacements)
			end
		end
		return self
	end
	
	# This method searches for a target function name and replaces
	# it with another name. Both the target and the replacement
	# name are given as arguments to the method. The method walks
	# through the node and checks whether:
	# * The node's class is a function definition or declaration
	# * The node's name is equal to the target node's name
	# If the checks are successful, the node's name will be replaced
	# The method will continue searching for other occurrences of
	# functions with the same name.
	#
	# The method returns itself.
	def seach_and_replace_function_definition(old_name,new_name)
		self.preorder do |node|
			if (node.function_definition? || node.function_declaration?) && (node.name == old_name)
				node.name = new_name
			end
		end
		return self
	end
	
	# This method is a small helper function to remove index
	# nodes from a node. It first clones to original node in
	# order to not overwrite it, then walks the node and removes
	# index nodes. Finally, it returns a new node.
	def remove_index
		node_clone = self.clone
		node_clone.preorder do |node|
			node.index.detach if node.index?
		end
		return node_clone
	end
	
	# This method is a small helper function which simply strips
	# any outer brackets from a node. If no outer brackets are
	# found, then nothing happens and the node itself is returned.
	def strip_brackets
		return (self.block?) ? self.stmts : self
	end
	
	# This method returns 'true' if the node is of the 'Variable'
	# class. Otherwise, it returns 'false'.
	def variable? ; (self.class == C::Variable) end
	
	# This method returns 'true' if the node is of the 'Array'
	# class. Otherwise, it returns 'false'.
	def array?
		return (self.class == C::Array)
	end
	
	# This method returns 'true' if the node is of the 'Pointer'
	# class. Otherwise, it returns 'false'.
	def pointer?
		return (self.class == C::Pointer)
	end
	
	# This method returns 'true' if the node is of the 'Parameter'
	# class. Otherwise, it returns 'false'.
	def parameter?
		return (self.class == C::Parameter)
	end
	
	# This method returns 'true' if the node is of the 'Declarator'
	# class. Otherwise, it returns 'false'.
	def declarator?
		return (self.class == C::Declarator)
	end
	
	# This method returns 'true' if the node is of the 'Declaration'
	# class. Otherwise, it returns 'false'.
	def declaration?
		return (self.class == C::Declaration)
	end
	
	# This method returns 'true' if the node is of the 'Index'
	# class. Otherwise, it returns 'false'.
	def index?
		return (self.class == C::Index)
	end
	
	# This method returns 'true' if the node is of the 'Call'
	# class. Otherwise, it returns 'false'.
	def call?
		return (self.class == C::Call)
	end
	
	# This method returns 'true' if the node is of the 'FunctionDef'
	# class. Otherwise, it returns 'false'.
	def function_definition?
		return (self.class == C::FunctionDef)
	end
	
	# This method returns 'true' if the node is of the 'Declarator'
	# class with its 'indirect_type' equal to 'Function' . Otherwise,
	# it returns 'false'.
	def function_declaration?
		return (self.class == C::Declarator && self.indirect_type.class == C::Function)
	end
	
	# This method returns 'true' if the node is of the 'Block'
	# class. Otherwise, it returns 'false'.
	def block?
		return (self.class == C::Block)
	end
	
	# This method returns 'true' if the node is of the 'For'
	# class. Otherwise, it returns 'false'.
	def for_statement?
		return (self.class == C::For)
	end
	
	# This method returns 'true' if the node is of the 'Less'
	# class. Otherwise, it returns 'false'.
	def less?
		return (self.class == C::Less)
	end
	
	# This method returns 'true' if the node is of the 'Add'
	# class. Otherwise, it returns 'false'.
	def add?
		return (self.class == C::Add)
	end
	
	# This method returns 'true' if the node is of the 'Subtract'
	# class. Otherwise, it returns 'false'.
	def subtract?
		return (self.class == C::Subtract)
	end
	
	# This method returns 'true' if the node is of the 'AddAssign'
	# class. Otherwise, it returns 'false'.
	def addassign?
		return (self.class == C::AddAssign)
	end
	
	# This method returns 'true' if the node is of the 'PostInc'
	# class. Otherwise, it returns 'false'.
	def postinc?
		return (self.class == C::PostInc)
	end
	
	# This method returns 'true' if the node is of the 'PreInc'
	# class. Otherwise, it returns 'false'.
	def preinc?
		return (self.class == C::PreInc)
	end
	
	# This method returns 'true' if the node is of the 'PostDec'
	# class. Otherwise, it returns 'false'.
	def postdec?
		return (self.class == C::PostDec)
	end
	
	# This method returns 'true' if the node is of the 'PreDec'
	# class. Otherwise, it returns 'false'.
	def predec?
		return (self.class == C::PreDec)
	end
	
	# This method returns 'true' if the node is of the 'IntLiteral'
	# class. Otherwise, it returns 'false'.
	def intliteral?
		return (self.class == C::IntLiteral)
	end
	
	# This method returns 'true' if the node is of the 'Assign'
	# class. Otherwise, it returns 'false'.
	def assign?
		return (self.class == C::Assign)
	end
	
	# This method returns 'true' if the node is of the 'Call'
	# class. Otherwise, it returns 'false'.
	def call?
		return (self.class == C::Call)
	end
	
	# This method returns 'true' if the node's class inherits
	# from the 'BinaryExpression' class. Otherwise, it returns
	# 'false'.
	def binary_expression?
		return (self.class.superclass == C::BinaryExpression)
	end
	
	# This method returns 'true' if the node's class inherits
	# from the 'AssignmentExpression' class. Otherwise, it returns
	# 'false'.
	def assignment_expression?
		return (self.class.superclass == C::AssignmentExpression)
	end
	
	# This method returns 'true' if the node is of the 'PostInc', 'PreInc'
	# class or if it is of the 'Assign' class and adds with a value of 1.
	# Otherwise, it returns 'false'.
	def unit_increment?
		return (self.class == C::PostInc) || (self.class == C::PreInc) || (self.class == C::Assign && self.rval.class == C::Add && self.rval.expr2.class == C::IntLiteral && self.rval.expr2.val == 1)
	end
	
	# This method returns 'true' if the node is performing an ALU
	# operation. Otherwise, it returns 'false'.
	def alu?
		return add? || subtract? || addassign? || postinc? || postdec? || preinc? || predec? || binary_expression?
	end
	
	# This method returns 'true' if the node is of the 'ExpressionStatement'
	# class. Otherwise, it returns 'false'.
	def statement?
		return (self.class == C::ExpressionStatement) || (self.class == C::Declaration)
	end
	
# From this point on are the private methods.
private
	
	# Override the existing 'indent' method to set the indent size
	# manually.
	def indent s, levels=1
		space = Bones::INDENT
		s.gsub(/^/, space)
	end
	
end

