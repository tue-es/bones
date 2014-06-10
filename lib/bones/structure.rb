
module Bones
	# This class represents a single structure in a species.
	# Such a structure is a single input or output 'structure'.
	# Stuctures can be set as part of array variables. Examples
	# of structures are given below:
	# 	0:9|element
	# 	0:0|shared
	# 	0:31,0:31|neighbourhood(-1:1,-1:1)
	# 	0:99,0:9|chunk(0:0,0:9)
	#
	class Structure < Common
		attr_reader :dimensions, :pattern, :parameters
		attr_accessor :name
		
		# The structure is initialized by the full name given as
		# a string. It is then analyzed and stored accordingly in
		# a number of class variables.
		def initialize(raw_data)
			data = raw_data.split(PIPE)
			pattern_data = data[1].split('(')
			dimension_data = data[0].split('[')
			@pattern = pattern_data[0].strip
			@name = (dimension_data.length == 2) ? dimension_data[0].strip : ''
			@dimensions = (dimension_data.length == 2) ? dimension_data[1].delete(']').split(DIM_SEP) : dimension_data[0].split(DIM_SEP)
			@parameters = (pattern_data.length > 1) ? pattern_data[1].delete(')').split(DIM_SEP) : []
		end
		
		# TODO: Implement the reverse function
		def reverse?
			true
		end
		
		# Method to find out if the structure has a parameter.
		# This is only the case if it is neighbourhood or chunk
		# based.
		def has_parameter?
			return (@parameters != [])
		end
		
		# Method to find out if the structure has a arrayname
		# defined. This is optional for a structure.
		def has_arrayname?
			return (@name != '')
		end
		
		# Method to get the start of a range given a dimension 'n'.
		# The method returns the proper simplified result, taking
		# chunk/neighbourhood-sizes into account.
		def from_at(n)
			if (neighbourhood?)
				return simplify('('+from(@dimensions[n])+')-('+from(@parameters[n])+')')
			else
				return simplify(from(@dimensions[n]))
			end
		end
		
		# Method to get the end of a range given a dimension 'n'.
		# The method returns the proper simplified result, taking
		# chunk/neighbourhood-sizes into account.
		def to_at(n)
			if (chunk?)
				return simplify('((('+to(@dimensions[n])+'+1)/('+to(@parameters[n])+'+1))-1)')
			elsif (neighbourhood?)
				return simplify('('+to(@dimensions[n])+')-('+to(@parameters[n])+')')
			else
				return simplify(to(@dimensions[n]))
			end
		end
		
		# Method to verify if a structure is empty or not (e.g. if
		# it is based on the 'void' pattern.
		def empty?
			return @pattern =~ /void/
		end
		
		# Method to check whether a structure is element-based.
		def element?
			return @pattern =~ /element/
		end
		
		# Method to check whether a structure is neighbourhood-based.
		def neighbourhood?
			return @pattern =~ /neighbourhood/
		end
		
		# Method to check whether a structure is chunk-based.
		def chunk?
			return @pattern =~ /chunk/
		end
		
		# Method to check whether a structure is shared-based.
		def shared?
			return @pattern =~ /shared/
		end
		
		# Method to check whether a structure is full-based.
		def full?
			return @pattern =~ /full/
		end
		
	end
	
end
