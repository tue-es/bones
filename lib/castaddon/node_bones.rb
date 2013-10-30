
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
	def preprocess(conditional=true)
		self.preorder do |node|
			if node.postinc? || node.preinc?
				node.replace_with(C::AssignmentExpression.parse(node.expr.to_s+' = '+node.expr.to_s+' + 1')) unless conditional && node.parent.for_statement?
			elsif node.postdec? || node.predec?
				node.replace_with(C::AssignmentExpression.parse(node.expr.to_s+' = '+node.expr.to_s+' - 1')) unless conditional && node.parent.for_statement?
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
		self.preorder do |node|
			if node.for_statement?
				from_statement = (node.init.assign?) ? node.init.rval : node.init.declarators[0].init
				from_loop = (from_statement.variable?) ? from_statement.name : from_statement.to_s
				to_loop = (node.cond.expr2.variable?) ? node.cond.expr2.name : ((node.cond.expr2.intliteral?) ? node.cond.expr2.val.to_s : node.cond.expr2.to_s)
				to_loop = to_loop.gsub(/\s/,'')
				to_loop = '('+to_loop+')-1' if node.cond.less?
				to_loop = simplify(to_loop)
				from_loop = simplify(from_loop)
				puts Bones::WARNING+'The loop iterator starts at: "'+from_loop+'" (expected "'+from+'")' if from_loop != from
				puts Bones::WARNING+'The loop iterator ends at: "'+to_loop+'" (expected "'+to+'")' if to_loop != to
				raise_error('The loop increment must be 1') if !(node.iter.unit_increment?)
				name = (node.init.assign?) ? node.init.lval.name : node.init.declarators.first.name
				return node.stmt, name
			end
		end
		raise_error('Unexpected number of for-loops')
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
	def search_and_replace_function_call(target,replacements)
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
	def search_and_replace_function_definition(old_name,new_name)
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
	
	# This method checks whether the given code has any conditional
	# statements (if-statements)
	def has_conditional_statements?
		self.preorder do |node|
			if node.if_statement?
				return true
			end
		end
		return false
	end
	
end

