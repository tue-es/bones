
module Bones
	# The species class contains 'algorithm classes', or 'species'.
	# Individual species contain a number of input and output
	# structures and possibly a prefix.
	#
	# Examples of species are found below:
	#
	# 	0:9|element -> 0:0|shared
	# 	0:31,0:31|neighbourhood(-1:1,-1:1) -> 0:31,0:31|element
	# 	unordered 0:99,0:9|chunk(0:0,0:9) -> 0:99|element
	#
	# == Naming conventions
	# The species class uses several naming conventions within
	# functions. They are as follows for the example species
	# '0:31,0:15|neighbourhood(-1:1,-1:1) ^ 0:31,0:15|element -> 0:31,0:15|element':
	#
	# input::        '0:31,0:15|neighbourhood(-1:1,-1:1) ^ 0:31,0:15|element'
	# output::       '0:31,0:15|element'
	# structures::   \['0:31,0:15|neighbourhood(-1:1,-1:1)', '0:31,0:15|element', '0:31,0:15|element']
	# structure::    '0:31,0:15|neighbourhood(-1:1,-1:1)' or '0:31,0:15|element' (twice)
	# pattern::      'neighbourhood' or 'element'
	# ranges::       \['0:31', '0:15'] or ['-1:1', '-1:1']
	# range::        '0:31' or '0:15' or '-1:1'
	# from::         '0' or '-1'
	# to::           '31' or '15' or '1'
	# sum::          '32' or '16' or '3'
	class Species < Common
		attr_reader :name, :inputs, :outputs, :prefix
		attr_accessor :skeleton_name, :settings
		
		# Initializes the species with a prefix, inputs and out-
		# puts. It additionally verifies the correctness of the 
		# provided raw data.
		def initialize(prefix, input, output)
			@prefix = prefix
			@name = (input+' '+ARROW+' '+output)
			@inputs = set_structures(input)
			@outputs = set_structures(output)
			@skeleton_name = nil
			@settings = nil
			self.verify_species
		end
		
		# This method splits the raw data (a string) into seperate
		# structures. The method returns an array of structures.
		def set_structures(raw_data)
			raw_data.split(WEDGE).map { |structure| Structure.new(structure) }
		end
		
		# Method to return an array of structures in a given di-
		# rection.
		def structures(direction)
			(direction == INPUT) ? @inputs : @outputs
		end
		
		# Method to return an array of structures for both direc-
		# tions.
		def all_structures
			@inputs + @outputs
		end
		
		# Method to return an ordered array of structures in a
		# given direction. The order is specified by an argument
		# which contains a list of pattern names.
		def ordered(direction,order)
			ordered = []
			order.each do |pattern_name|
				self.structures(direction).each do |structure|
					ordered.push(structure) if structure.pattern == pattern_name
				end
			end
			
			# Remove structures with a duplicate name (for matching only - and only if names are given)
			if ordered.all? { |s| s.name != "" }
				names = []
				ordered.each do |structure|
					ordered.delete(structure) if names.include?(structure.name)
					names.push(structure.name)
				end
			end
			return ordered
		end
		
		# This method maps an algorithm species to a skeleton.
		# This is done based on a mapping file provided as part
		# of the skeleton library. If multiple skeletons match
		# the current species, the first found match is taken.
		# This method does not return any values, but instead
		# sets the class variables +skeleton_name+ and +settings+.
		def set_skeleton(mapping_file)
			matches = []
			File.read(mapping_file).each_line do |line|
				next if line =~ /^#/
				data = line.split(/\s:/)
				matches.push(data) if match_species?(data[0].split(ARROW))
			end
			puts MESSAGE+'Multiple matches in skeleton file, selecting the first listed' if matches.length > 1
			if matches.length != 0
				@skeleton_name = matches[0][1].delete(':').strip
				@settings = matches[0][2].delete(':').strip
			end
		end
		
		# This method is called by the +set_skeleton+ method. It
		# performs a match between the current species and the
		# species found in the mapping file. The matching is based
		# on a fixed order of patterns. The method returns either
		# true or false.
		def match_species?(file_data)
			DIRECTIONS.each_with_index do |direction,num_direction|
				file_structures = file_data[num_direction].split(WEDGE)
				counter = 0
				search_structures = ordered(direction,['chunk','neighbourhood','element','shared','void'])
				search_structures.each do |search_structure|
					if !match?(file_structures[counter],search_structure)
						if (counter != 0) && (file_structures[counter-1] =~ /\+/) && match?(file_structures[counter-1],search_structure)
							counter = counter - 1
						else
							return false
						end
					end
					counter = counter + 1
				end
				return false if counter != file_structures.length
			end
			return true
		end
		
		# This method implements the match between a species'
		# structure and a structure found in the mapping file.
		# It is called from the +match_species+ method. It first
		# checks for a pattern match, followed by a match of the
		# dimensions. The method returns either true or false.
		# TODO: Complete the matching (N-dimensional).
		def match?(file,search)
			if (file =~ /#{search.pattern}/)
				condition = true
				
				# Check for parameters
				if file.split('(').length == 2
					parameters = file.split('(')[1].split(')')[0]
					if (parameters == 'D') ||
						 ((parameters == 'N') && (search.parameters.length == 1)) ||
						 ((parameters == 'N,N') && (search.parameters.length == 2)) ||
						 ((parameters == 'N,1') && (search.parameters.length == 2) && simplify(sum(search.parameters[1])) == '1') ||
						 ((parameters == '1,N') && (search.parameters.length == 2) && simplify(sum(search.parameters[0])) == '1') ||
						 ((parameters == search.parameters.map { |r| simplify(sum(r)) }.join(',')))
						condition = condition && true
					else
						condition = false
					end
				end
				
				# Check for dimensions
				dimensions = file.split(PIPE)[0].strip
				if (dimensions == 'D') ||
					 ((dimensions == 'N') && (search.dimensions.length == 1)) ||
					 ((dimensions == 'N,N') && (search.dimensions.length == 2)) ||
					 ((dimensions == search.dimensions.map { |r| simplify(sum(r)) }.join(',')))
					condition = condition && true
				else
					condition = false
				end
				
				# Return
				return true if condition == true
			end
			return false
		end
		
		# Method to verify whether this species is shared-based
		# or not. The method either returns true (if 'shared' is
		# included in the input or output) or false (if not).
		def shared?
			(@name =~ /shared/)
		end
		
		# This method verifies if the species dimensions match
		# with its parameters in terms of number of dimensions.
		def verify_species
			DIRECTIONS.each do |direction|
				structures(direction).each do |structure|
					if (structure.has_parameter?) && (structure.dimensions.length != structure.parameters.length)
						puts WARNING+'Parameter dimension mismatch: '+structure.parameters.inspect+' versus '+structure.dimensions.inspect
					end
					structure.dimensions.each do |dimension|
						puts WARNING+'Negative range given: '+dimension.inspect if (simplify(sum(dimension)).to_i < 1) && !(sum(dimension) =~ /[a-zA-Z]/)
					end
				end
			end
		end
		
	end
	
end

