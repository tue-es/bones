# This class provides an extention to the CAST index class.
# The class contains a number of functions applicable to array
# accesses of the form 'array[x][y]' or 'vector[i]'.
#
# The provided methods are helpers to extend the CAST functionality
# and to clean-up the Bones classes.
class C::Index
	
	# This method is a recursive method which *gets* the name of
	# a variable from the index definition. Depending on the
	# number of dimensions, it will go deeper into the structure
	# and eventually return the name.
	def variable_name
		return (self.expr.variable?) ? self.expr.name : self.expr.variable_name
	end
	
	# This method is a recursive method which *sets* the name of
	# a variable from the index definition. Depending on the
	# number of dimensions, it will go deeper into the structure
	# and eventually set the name.
	def variable_name=(name)
		(self.expr.variable?) ? self.expr.name = name : self.expr.variable_name=(name)
	end
	
	# This method returns the dimension of an index expression.
	# It starts at dimension 1, but if it can find a new dimension
	# it will increment the count and call itself again.
	def dimension(count=1)
		return (self.expr.index?) ? self.expr.dimension(count+1) : count
	end
	
	# This method returns the index itself at a given dimension.
	# It uses recursion to iterate through the dimensions, but
	# will eventually return a new index node.
	def index_at_dimension(dimension)
		return (dimension == 0) ? self.index : self.expr.index_at_dimension(dimension-1)
	end
	
end

