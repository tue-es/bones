

# This is an extension to the CAST Node class. This particular extension is for
# A-Darwin only methods. These methods are mainly used to extract loops and loop
# data from the CAST nodes.
class C::Node
	
	# This method retrieves all directly following loops from a node, i.e. the
	# loops belonging to a perfectly nested loop. It is a recursive method: it
	# retrieves a first loop and calls the method again on the body of the loop.
	# It collects all the data in the +loop_data+ array.
	def get_direct_loops(loop_data = [])
		
		# Retrieve the next loop
		new_loop = get_single_loop()
		
		# Don't continue if the loop is independent of the writes in the code. This
		# is part of the selection process of what loops to be considered inner or
		# outer loops.
		if loop_data.length > 0
			written_indices = self.clone.get_accesses().map do |a|
				(a[:type] == "write") ? a[:indices].map{ |i| i.to_s } : []
			end
			if !written_indices.flatten.uniq.include?(new_loop[:var])
				return loop_data
			end
		end
		
		# Push the new loop into the array
		loop_data.push(new_loop)
		
		# Check whether the current is actually a loop.
		# TODO: Is this check really needed or is this just a safety net?
		if self.for_statement? && self.stmt
			body = self.stmt.stmts
			
			# Check whether or not there is another loop directly following and make
			# sure that the body is not empty.
			if body.length == 1 && body.first.for_statement?
				body.first.get_direct_loops(loop_data)
			end
		end
		
		# Return all the loop data
		return loop_data
	end
	
	# This method retrieves all array references in the current node. It retrieves
	# information on loops and on if-statements as well. This method is
	# destructive on the current node. It is furthermore called recursively.
	def get_accesses(accesses = [],loop_data = [],if_statements = [])
		
		# Iterate over all the nodes
		self.preorder do |node|
			
			# A loop has been found. Proceed as follows: 1) store the loop data, 2)
			# call this method recursively on the loop body, and 3) remove the loop
			# body from the node.
			if node.for_statement? && node.stmt
				next_loop_data = loop_data.clone
				next_loop_data.push(node.get_single_loop)
				node.stmt.clone.get_accesses(accesses,next_loop_data,if_statements)
				node.remove_node(node.stmt)
			
			# An if-statement has been found. Proceed as follows: 1) store the (one or
			# more) if-statement conditions, 2) call this method recursively on the
			# if-statement body, and 3) remove the if-statement body from the node.
			elsif node.if_statement?
				next_if_statements = if_statements.clone
				node.cond.get_conditions().each do |condition|
					next_if_statements.push(condition)
				end
				node.then.clone.get_accesses(accesses,loop_data,next_if_statements)
				node.remove_node(node.then)
			
			# We haven't found an if-statement or loop in the current node, so it
			# implies that we can search for an array reference.
			# TODO: Array references as part of conditions or bounds of loops are not
			# found in this way.
			else
			
				# Collect all the writes we have seen so far. This is used to check for
				# references that are 'register' references because they have been
				# written before.
				writes = accesses.map{ |e| e[:access] if e[:type] == 'write' }.flatten
				
				# Collect the potential references
				to_search = []
				if node.assignment_expression?
					to_search << [node.lval,'write',true]
					to_search << [node.lval,'read',false] if !node.assign?
					to_search << [node.rval,'read',false]
				elsif node.binary_expression?
					to_search << [node.expr1,'read',false]
					to_search << [node.expr2,'read',false]
				elsif node.declarator? && node.init
					to_search << [node.init,'read',false]
				end
				
				# Process the potential references into 'accesses' hashes
				to_search.each do |item|
					if item[2] || (writes & item[0].get_index_nodes().flatten).empty?
						item[0].get_index_nodes().each do |access|
							accesses << {
								:access => access,
								:name => access.get_array_name(),
								:indices => access.get_indices(),
								:type => item[1],
								:loop_data => loop_data,
								:if_statements => if_statements
							}
						end
					end
				end
			end
		end
		
		# Return the array references as hashes
		return accesses.uniq
	end
	
	# This method retrieves the bounds for an if-statement. The method is called
	# recursively if there are multiple conditions.
	# TODO: What about '||' (or) conditions? They are currently handles as '&&'.
	# TODO: Are these all the possibilities (&&,||,>=,>,<=,<) for conditions?
	def get_conditions(results=[])
		
		# Recursive call for 'And' (&&) and 'or' (||) compound conditions
		if and? || or?
			expr1.get_conditions(results)
			expr2.get_conditions(results)
			
		# Greater than or equal (>=)
		elsif more_or_equal?
			results << [simplify("#{expr1}")+'='+simplify("(#{expr2})"),'']
			
		# Greater than (>)
		elsif more?
			results << [simplify("#{expr1}")+'='+simplify("(#{expr2})+1"),'']
			
		# Less than or equal (<=)
		elsif less_or_equal?
			results << ['',simplify("#{expr1}")+'='+simplify("(#{expr2})")]
			
		# Less than (<)
		elsif less?
			results << ['',simplify("#{expr1}")+'='+simplify("(#{expr2})-1")]
			
		# Equal (==)
		elsif equality?
			results << ['','']#[simplify("#{expr1}"),simplify("(#{expr2})")]

		# Unsupported conditions
		else
			raise_error("Unsupported if-condition: #{self.to_s}")
		end
	end
	
	# This method retrieves a single loop from the current node and collects its
	# data: 1) the loop variable, 2) the lower-bound, 3) the upper-bound, and 4)
	# the loop step.
	# FIXME: For decrementing loops, should the min/max be swapped?
	def get_single_loop()
		loop_datum = { :var => '', :min => '', :max => '', :step => ''}
		if self.for_statement?
			
			# Get the loop start condition and the loop variable.
			# TODO: Add support for other types of initialisations, e.g. a declaration
			if self.init.assign?
				loop_datum[:var] = self.init.lval.name
				loop_datum[:min] = self.init.rval.get_value.to_s
			elsif self.init.declaration?
				loop_datum[:var] = self.init.declarators.first.name
				loop_datum[:min] = self.init.declarators.first.init.to_s
			else
				raise_error("Unsupported loop initialization: #{self.init}")
			end
			
			# Get the loop's upper-bound condition.
			# TODO: Add support for the unsupported cases.
			var_is_on_left = (self.cond.expr1.get_value == loop_datum[:var])
			loop_datum[:max] = case
				when self.cond.less? then (var_is_on_left) ? simplify("#{self.cond.expr2.get_value}-1") : "unsupported"
				when self.cond.more? then (var_is_on_left) ? "unsupported" : simplify("#{self.cond.expr1.get_value}-1")
				when self.cond.less_or_equal? then (var_is_on_left) ? "#{self.cond.expr2.get_value}" : "unsupported"
				when self.cond.more_or_equal? then (var_is_on_left) ? "unsupported" : "#{self.cond.expr1.get_value}"
			end
			raise_error("Unsupported loop condition: #{self.cond}") if loop_datum[:max] == "unsupported"
			
			# Get the loop iterator.
			# TODO: Investigate whether we can handle non-basic cases
			iterator = self.iter.to_s
			loop_datum[:step] = case iterator
				when "#{loop_datum[:var]}++" then '1'
				when "++#{loop_datum[:var]}" then '1'
				when "#{loop_datum[:var]}--" then '-1'
				when "--#{loop_datum[:var]}" then '-1'
				else simplify(self.iter.rval.to_s.gsub(loop_datum[:var],'0'))
			end
		end
		return loop_datum
	end
	
	# This method retrieves all loops from a loop nest. The method is based on the
	# +get_single_loop+ method to extract the actual loop information.
	def get_all_loops()
		loops = []
		self.preorder do |node|
			loops << node.get_single_loop() if node.for_statement?
		end
		return loops
	end
	
	# This method retrieves all nodes from the current node that are index node.
	# Such nodes represent array references, e.g. in A[i+3], [i+3] is the index
	# node.
	def get_index_nodes()
		nodes = []
		self.preorder do |node|
			nodes << node if node.index? && !node.parent.index?
		end
		return nodes
	end
	
	# This method retrieves all indices of index nodes from the current node.
	def get_indices()
		indices = []
		self.postorder do |node|
			indices << node.index if node.index?
		end
		return indices
	end
	
	# This method retrieves the name of the array reference.
	def get_array_name()
		self.preorder do |node|
			return node.expr.to_s if node.index? && !node.expr.index?
		end
	end
	
	# This method retrieves all variable declarations
	def get_var_declarations()
		vars = []
		self.preorder do |node|
			if node.declaration?
				node.declarators.each do |decl|
					vars << decl.name
				end
			end
		end
		return vars
	end
	
	# This method retrieves the value from the current node. The value can be an
	# integer (in case of a constant) or a string (in case of a variable).
	def get_value()
		return self.val if self.intliteral?
		return self.name if self.variable?
		return self.to_s
	end
	
end