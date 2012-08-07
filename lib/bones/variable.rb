
module Bones
	# This class represents individual variables. Variables have a
	# name, a type, a direction and an id. They might also have an
	# algorithmic species coupled if they represent an array. The 
	# variable type is in AST form and holds information which can
	# be extracted through methods implemented for the Type class.
	# 
	# The provided methods are able to obtain information from the
	# variables, such as the number of dimensions and the definition.
	# Other methods query variables for properties, such as whether
	# a variable is an array or not. Several methods should only be
	# executed if the variable is an array - this is not checked
	# within the methods itself.
	class Variable < Common
		attr_reader :name, :type, :factors, :direction
		attr_accessor :species, :size, :guess
		
		# Method to initilize the variable class with a name, type
		# and a direction.
		def initialize(name,type,size,direction,id,shared)
			@name      = name
			@type      = type
			@size      = size
			@direction = direction
			@id        = id
			@shared    = shared
			@guess     = false
			@species   = nil
		end
		
		# This method returns the device name of the variable.
		def device_name
			DEVICE+@name
		end
		
		# This method returns the 'golden' name of a variable.
		# This is used to generate verification code.
		def golden_name
			GOLDEN + '_' + @name + '_' + @id
		end
		
		# Method to find out whether a variable is used as an in-
		# put. If the current algorithm uses the 'shared' pattern,
		# then the variable must be input only, otherwise it can
		# be both input or inout to be considered input. The return
		# value is boolean.
		def input?
			return (@shared) ? (@direction == INPUT) : (@direction == INPUT || @direction == INOUT)
		end
		
		# Method to find out whether a variable is used as an out-
		# put. The variable can be both output or inout to be con-
		# sidered input. The return value is boolean.
		def output?
			return (@direction == OUTPUT || @direction == INOUT)
		end
		
		# Method to obtain the type of a variable, omitting any
		# information about arrays or pointers. Examples of out-
		# puts are: +int+, +float+ or +char+. This functionality
		# is implemented as a recursive search in the Type class,
		# since the type is in AST form.
		def type_name
			@type.type_name.to_s
		end
		
		# Method to obtain the number of dimensions of a variable.
		# This method returns a positive integer. The functionality
		# is implemented as a recursive search in the Type class.
		def dimensions
			@type.dimensions
		end
		
		# Method to return the device version of a variable's
		# definition. This includes the variable's type, zero or
		# more stars and the variable's name, all returned as a
		# single string.
		def device_definition
			type_name + device_pointer + ' ' + @name
		end
		
		# Method to obtain the full defintion of a variable. This
		# includes the variable type, the variable name, and the
		# dimensions and/or pointers. Example return values are:
		# 	int example[10]
		# 	char **variable
		# 	unsigned int* example[N]
		# 	float array[][]
		def definition
			definition_prefix + ' ' + @name + definition_suffix
		end
		
		# This method returns a star ('*') if the variable is an
		# array or an empty string if it is not. It will not be
		# able to return more than a single star, as device varia-
		# bles are assumed to be flattened.
		def device_pointer
			(@type.array_or_pointer?) ? '*' : ''
		end
		
		# Method to flatten an array into a one dimensional array.
		# If an array has multiple dimensions, the method will
		# return a '[0]' for each additional dimension. This method
		# assumes that the variable is an array.
		def flatten
			''+'[0]'*(dimensions-1)
		end
		
		# This method returns the initialization code for a varia-
		# ble, formatted as a string. This is used to generate the
		# verification code.
		def initialization
			(dynamic?) ? " = (#{definition_prefix})malloc(#{@size.join('*')}*sizeof(#{type_name}));"+NL : ';'+NL
		end
		
		# This method detects whether the variable is dynamically
		# allocated or not. It returns either true or false.
		def dynamic?
			(definition_prefix.count('*') > 0)
		end
		
		# Method to return an array of multiplication factors for
		# multi-dimensional arrays that need to be flattened. For
		# every dimension, one factor is generated.
		def set_factors
			raise_error("Species dimensions (#{@species.dimensions.length}) and array dimensions (#{@size.length}) mismatch for array '#{@name}'") if @species.dimensions.length != @size.length
			sizes, @factors = [], []
			@species.dimensions.each_with_index do |dimension,num_dimension|
				(sizes.empty?) ? @factors.push('') : @factors.push('*'+sizes.join('*'))
				sizes.push(simplify(@size[dimensions-num_dimension-1]))
			end
		end
		
		# Method to return the full flattened address.
		def flatindex
			indices = []
			@species.dimensions.each_with_index do |dimension,num_dimension|
				index_reverse = !(@species.reverse?) ? num_dimension : @species.dimensions.length-num_dimension-1 # FIXME: REVERSED
				if (from(dimension) != to(dimension))
					data = "#{GLOBAL_ID}_#{index_reverse}"
				else
					data = from(dimension)
				end
				indices.push(data + @factors[index_reverse])
			end
			return indices.join(' + ')
		end
		
	# Start of the class's private methods.
	private
		
		# Method to obtain the prefix of a variable's definition,
		# formatted as a string. The string contains the variable's
		# type and zero or more stars ('*').
		def definition_prefix
			@type.to_s.partition('[')[0].strip
		end
		
		# Method to obtain the suffix of a variable's definition,
		# formatted as a string. The string is either empty or
		# contains one or more brackets (e.g. [N][M] or [10]).
		def definition_suffix
			@type.to_s.partition('[')[1]+@type.to_s.partition('[')[2]
		end
		
	end
	
end
