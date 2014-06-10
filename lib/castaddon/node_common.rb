
# This file provides the common methods that extend the CAST node
# class. These methods are used by both Bones and A-Darwin.
class C::Node
	
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
	def search_and_replace_node(target,replacements)
		if (self.class == target.class) && (self.match?(target))
			return replacements
		end
		self.preorder do |node|
			if (node.class == target.class) && (node.match?(target))
				if (node.parent != nil)
					node.replace_with(replacements)
				else
					return replacements
				end
				return self
			end
		end
		return self
	end
	
	# This method is a small helper function to remove a node once.
	def remove_once(target)
		self.postorder do |node|
			if node == target
				node.detach
				return self
			end
		end
		return self
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
	
	# This method returns 'true' if the node is of the 'If'
	# class. Otherwise, it returns 'false'.
	def if_statement?
		return (self.class == C::If)
	end
	
	# This method returns 'true' if the node is of the 'And'
	# class. Otherwise, it returns 'false'.
	def and?
		return (self.class == C::And)
	end
	
	# This method returns 'true' if the node is of the 'Or'
	# class. Otherwise, it returns 'false'.
	def or?
		return (self.class == C::Or)
	end
	
	# This method returns 'true' if the node is of the 'Equal'
	# class. Otherwise, it returns 'false'.
	def equality?
		return (self.class == C::Equal)
	end
	
	# This method returns 'true' if the node is of the 'Less'
	# class. Otherwise, it returns 'false'.
	def less?
		return (self.class == C::Less)
	end
	
	# This method returns 'true' if the node is of the 'More'
	# class. Otherwise, it returns 'false'.
	def more?
		return (self.class == C::More)
	end
	
	# This method returns 'true' if the node is of the 'LessOrEqual'
	# class. Otherwise, it returns 'false'.
	def less_or_equal?
		return (self.class == C::LessOrEqual)
	end
	
	# This method returns 'true' if the node is of the 'MoreOrEqual'
	# class. Otherwise, it returns 'false'.
	def more_or_equal?
		return (self.class == C::MoreOrEqual)
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
	
	# This method returns 'true' if the node is of the 'StringLiteral'
	# class. Otherwise, it returns 'false'.
	def string?
		return (self.class == C::StringLiteral)
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
		space = INDENT
		s.gsub(/^/, space)
	end
	
end

