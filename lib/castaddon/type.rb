module C
	# This class provides an extention to the CAST type class. It
	# contains a number of functions applicable to types such as
	# pointers, arrays, structures, floats, integers, etc.
	#
	# The provided methods are just helpers to extend the CAST
	# functionality and to clean-up the Bones classes.
	class Type
		
		# This method is used to determine whether the variable is
		# an array and/or a pointer. Returns either true or false.
		def array_or_pointer?
			((self.class == C::Array) || (self.class == C::Pointer))
		end
		
		# This method recursively searches for the type of a variable.
		# Recursion is needed when a type is an array or a pointer.
		# The method eventually returns one of the CAST algorithm
		# types being either: void, int, float, char, bool, complex
		# or imaginary.
		def type_name
			(self.array_or_pointer?) ? self.type.type_name : self
		end
		
		# This method returns the variable's dimension as an integer.
		# it uses recursion in case the type is an array or a pointer.
		# Types that are neither arrays nor pointers have a dimension
		# of zero. For arrays and pointers, each '*' or '[]' contributes
		# to one additional dimension.
		def dimensions(count=0)
			(self.array_or_pointer?) ? self.type.dimensions(count+1) : count
		end
		
	end
	
end

