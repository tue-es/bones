
module Bones
	# This is the C99 pre-processor for Bones. It has two tasks:
	# * To remove all lines starting with '#' (pre-processor directives).
	# * To detect all pragma's forming algorithm classes from the source.
	#
	# ==== Attributes:
	# * +header_code+ - All the code that was removed by the pre-processor but was not relevant to Bones. This contains for example includes and defines.
	# * +algorithms+ -  An array of identified algorithms, each of class Bones::Algorithm.
	# * +target_code+ - The processed code containing no Bones directives nor other pre-processor directives (such as includes and defines).
	class Preprocessor < Common
		attr_reader :header_code, :algorithms, :target_code, :device_header, :defines
		
		# Denotes the start of an algorithmic species.
		IDENTIFIER = '#pragma species'
		
		# Regular expression to identify whitespaces (tabs, spaces).
		WHITESPACE = '\s*'
		
		# This directive denotes the start of a algorithm. It is based on the IDENTIFIER constant.
		PRIMITIVE_START = IDENTIFIER+' kernel'
		# This directive denotes the end of a algorithm. It is based on the IDENTIFIER constant.
		PRIMITIVE_END = IDENTIFIER+' endkernel'
		
		# A regular expression captures a prefix in a algorithm (e.g. unordered/multiple).
		REGEXP_PREFIX = /^[a-z]+ /
		
		# Providing a default name in case a algorithm is not named.
		DEFAULT_NAME = 'algorithm'
		
		# This is the method which initializes the preprocessor.
		# Initialization requires the target source code to process,
		# which is then set as the class variable +@source_code+.
		def initialize(source_code,directory,filename)
			@source_code = source_code
			@target_code = ''
			@header_code = ''
			@device_header = ''
			@directory = directory
			@filename = filename
			@algorithms = Array.new
			@defines = {}
			@found_algorithms = 0
		end
		
		# This is the method to perform the actual preprocessing.
		# This method takes care of all the pre-processor tasks.
		# The output is stored in the three attributes +header_code+,
		# +algorithms+, and +target_code+.
		def process
			algorithm_code = ''
			species = nil
			found = 0
			
			# Process the file line by line
			@source_code.each_line.with_index do |line,index|
				if line =~ /^#{WHITESPACE}#/
					
					# Keep 'include' statements as header code
					if line =~ /^#{WHITESPACE}#include/
						@header_code += line
						if line =~ /"(.*)"/
							process_header($1)
						end
					
					# Process 'define' statements for the algorithm code, but also keep as header code
					elsif line =~ /^#{WHITESPACE}#define/
						@header_code += line
						@device_header += line
						match = line.split(/\/\//)[0].scan(/^#{WHITESPACE}#define\s+(\w+)\s+(\S*)/)
						@defines[match.first[0].to_sym] = match.first[1]
					
					# Found the start of algorithm marker
					elsif line =~ /^#{WHITESPACE}#{PRIMITIVE_START}/
						if found == 0
							line = replace_defines(line,@defines)
							prefix, input, output = marker_to_algorithm(line)
							puts MESSAGE+'Found algorithm "'+(prefix+' '+input+' '+ARROW+' '+output).lstrip+'"' if VERBOSE
							species = Bones::Species.new(prefix,input,output)
							@found_algorithms = @found_algorithms + 1
						end
						found = found + 1
					
					# Found the end of algorithm marker
					elsif line =~ /^#{WHITESPACE}#{PRIMITIVE_END}/
						if found == 1
							name = line.strip.scan(/^#{WHITESPACE}#{PRIMITIVE_END} (.+)/).join
							name = DEFAULT_NAME if name == ''
							@algorithms.push(Bones::Algorithm.new(name,@filename,index.to_s,species,algorithm_code))
							algorithm_code = ''
						end
						found = found - 1
					end
				else
					if found > 0
						algorithm_line = replace_defines(line,@defines)
						@target_code += algorithm_line
						algorithm_code += algorithm_line if line !~ /^#{WHITESPACE}#/
					else
						@target_code += line
					end
				end
			end
			puts WARNING+'Begin/end kernel mismatch ('+@found_algorithms.to_s+' versus '+@algorithms.length.to_s+'), probably missing a "'+PRIMITIVE_END+'"' unless @algorithms.length == @found_algorithms
		end
		
		# This is the method to preprocess a header file. Currently,
		# it only searches for defines and adds those to a list. In
		# the meanwhile, it also handles ifdef's.
		def process_header(filename)
			ifdefs = [true]
			
			# Process the file line by line
			File.read(File.join(@directory,filename)).each_line.with_index do |line,index|
				if line =~ /^#{WHITESPACE}#/
					
					# Process 'include' statements
					if line =~ /^#{WHITESPACE}#include/ && ifdefs.last
						if line =~ /"(.*)"/
							process_header($1)
						end
					
					# Process 'define' statements
					elsif line =~ /^#{WHITESPACE}#define/ && ifdefs.last
						match = line.split(/\/\//)[0].scan(/^#{WHITESPACE}#define\s+(\w+)\s+(\S*)/)
						@defines[match.first[0].to_sym] = match.first[1].strip
					
					# Process 'ifdef' statements
					elsif line =~ /^#{WHITESPACE}#ifdef#{WHITESPACE}(\w+)/
						valid = (ifdefs.last) ? @defines.has_key?($1.to_sym) : false
						ifdefs.push(valid)
						
					# Process 'endif' statements
					elsif line =~ /^#{WHITESPACE}#endif/
						ifdefs.pop
					end
				end
			end
		end
		
	# From this point on are the private methods.
	private
		
		# Method to extract the algorithm details from a marker found in code.
		def marker_to_algorithm(marker)
			algorithm = marker.strip.scan(/^#{WHITESPACE}#{PRIMITIVE_START} (.+)/).join
			prefix = ''
			if algorithm =~ REGEXP_PREFIX
				split = algorithm.partition(' ')
				prefix = split[0]
				algorithm = split[2]
			end
			input  = algorithm.split(ARROW)[0].strip
			output = algorithm.split(ARROW)[1].strip
			return prefix, input, output
		end
		
	end
	
end

