
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
		attr_reader :header_code, :algorithms, :target_code, :device_header, :defines, :scop, :copies
		
		# Denotes the start of an algorithmic species.
		IDENTIFIER = '#pragma species'
		
		# Regular expression to identify whitespaces (tabs, spaces).
		WHITESPACE = '\s*'
		
		# This directive denotes the start of a algorithm. It is based on the IDENTIFIER constant.
		SPECIES_START = IDENTIFIER+' kernel'
		# This directive denotes the end of a algorithm. It is based on the IDENTIFIER constant.
		SPECIES_END = IDENTIFIER+' endkernel'
	
		# Start of the scop
		SCOP_START = '#pragma scop'
		# Enf of the scop
		SCOP_END = '#pragma endscop'
		
		# Synchronise directive.
		SYNC = IDENTIFIER+' sync'
		
		# Copy in directive.
		COPYIN = IDENTIFIER+ ' copyin'
		# Copy out directive.
		COPYOUT = IDENTIFIER+ ' copyout'
		
		# A regular expression captures a prefix in a algorithm (e.g. unordered/multiple).
		REGEXP_PREFIX = /^[a-z]+ /
		
		# Providing a default name in case a algorithm is not named.
		DEFAULT_NAME = 'algorithm'
		
		# This is the method which initializes the preprocessor.
		# Initialization requires the target source code to process,
		# which is then set as the class variable +@source_code+.
		def initialize(source_code,directory,filename,scheduler)
			@source_code = source_code
			@target_code = []
			@header_code = ''
			@device_header = ''
			@directory = directory
			@filename = filename
			@algorithms = Array.new
			@copies = Array.new
			@defines = {}
			@found_algorithms = 0
			@scheduler = scheduler
		end
		
		# This is the method to perform the actual preprocessing.
		# This method takes care of all the pre-processor tasks.
		# The output is stored in the three attributes +header_code+,
		# +algorithms+, and +target_code+.
		def process
			algorithm_code = ''
			species = nil
			found = 0
			alloc_index, free_index = 0, 0
			block_comment = false
			a_scop_was_found = false
			
			# Process the file line by line
			@source_code.each_line.with_index do |line,index|

				# Don't consider one-line comments
				if !(line =~ /^#{WHITESPACE}\/\//)

					# Found the start of a block comment
					if line =~ /\/\*/
						block_comment = true
					end

					# Search for the end of the block comment
					if block_comment
						if line =~ /\*\//
							block_comment = false
						end
						@target_code << line

					# Not in a block-comment
					else

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
							elsif line =~ /^#{WHITESPACE}#{SPECIES_START}/
								if found == 0
									line = replace_defines(line,@defines)
									prefix, input, output = marker_to_algorithm(line)
									puts MESSAGE+'Found algorithm "'+(prefix+' '+input+' '+ARROW+' '+output).lstrip+'"' if VERBOSE
									species = Bones::Species.new(prefix,input,output)
									@found_algorithms = @found_algorithms + 1
								end
								found = found + 1
								#@target_code << "int bones_temp_species_start = '#{line.gsub(NL,'')}';"+NL
							
							# Found the end of algorithm marker
							elsif line =~ /^#{WHITESPACE}#{SPECIES_END}/
								if found == 1
									name = line.strip.scan(/^#{WHITESPACE}#{SPECIES_END} (.+)/).join
									name = DEFAULT_NAME if name == ''
									@algorithms.push(Bones::Algorithm.new(name,@filename,index.to_s,species,algorithm_code))
									algorithm_code = ''
								end
								found = found - 1
								#@target_code << "int bones_temp_species_end = '#{line.gsub(NL,'')}';"+NL
								
							# Found a sync marker
							elsif @scheduler && line =~ /^#{WHITESPACE}#{SYNC}/
								sync = line.strip.scan(/^#{WHITESPACE}#{SYNC} (.+)/).join
								@target_code << "bones_synchronize(#{sync});"+NL
								
							# Found a copyin marker
							elsif @scheduler && line =~ /^#{WHITESPACE}#{COPYIN}/
								copies = line.strip.scan(/^#{WHITESPACE}#{COPYIN} (.+)/).join.split(WEDGE).map{ |c| c.strip }
								copies.each_with_index do |copy,copynum|
									name = copy.split('[').first
									domain = copy.scan(/\[(.+)\]/).join.split(DIM_SEP)
									deadline = copy.split('|').last
									@copies.push(Bones::Copy.new(name,domain,deadline,'in',"#{index*100+copynum}"))
									@target_code << "bones_copyin_#{index*100+copynum}_#{name}(#{name});"+NL
								end
								
							# Found a copyout marker
							elsif @scheduler && line =~ /^#{WHITESPACE}#{COPYOUT}/
								copies = line.strip.scan(/^#{WHITESPACE}#{COPYOUT} (.+)/).join.split(WEDGE).map{ |c| c.strip }
								copies.each_with_index do |copy,copynum|
									name = copy.split('[').first
									domain = copy.scan(/\[(.+)\]/).join.split(DIM_SEP)
									deadline = copy.split('|').last
									@copies.push(Bones::Copy.new(name,domain,deadline,'out',"#{index*100+copynum}"))
									@target_code << "bones_copyout_#{index*100+copynum}_#{name}(#{name});"+NL
								end
							end
							
							# Check if it was a 'pragma scop' / 'pragma endscop' line
							if line =~ /^#{WHITESPACE}#{SCOP_START}/
								alloc_index = index
								a_scop_was_found = true
							elsif line =~ /^#{WHITESPACE}#{SCOP_END}/
								free_index = @target_code.length
							end
							
						else
							if found > 0
								algorithm_line = replace_defines(line,@defines)
								@target_code << algorithm_line
								algorithm_code += algorithm_line if line !~ /^#{WHITESPACE}#/
							else
								@target_code << line
							end
						end
					end
				else
					@target_code << line
				end
			end
			puts WARNING+'Begin/end kernel mismatch ('+@found_algorithms.to_s+' versus '+@algorithms.length.to_s+'), probably missing a "'+SPECIES_END+'"' unless @algorithms.length == @found_algorithms
			
			# Add frees and mallocs
			if @scheduler
				alloc_code, free_code = '', ''
				included_copies = []
				copies.each do |copy|
					if !included_copies.include?(copy.name)
						alloc_code += copy.get_function_call('alloc')+NL
						free_code += copy.get_function_call('free')+NL
						included_copies << copy.name
					end
				end
			end
			
			# Add timers (whole scop timing) and frees/mallocs to the code
			if a_scop_was_found
				offset = @header_code.lines.count
				@target_code.insert(alloc_index-offset, 'bones_timer_start();'+NL)
				if @scheduler
					@target_code.insert(alloc_index-offset+1, alloc_code)
					@target_code.insert(free_index+2, free_code)
					@target_code.insert(free_index+3, 'bones_timer_stop();'+NL)
				else
					@target_code.insert(free_index+2, 'bones_timer_stop();'+NL)
				end
			else
				puts WARNING+'No "#pragma scop" and "#pragma endscop" found!'
			end
			
			# Join the array
			@target_code = @target_code.join('')
		end
		
		# This is the method to preprocess a header file. Currently,
		# it only searches for defines and adds those to a list. In
		# the meanwhile, it also handles ifdef's.
		def process_header(filename)
			ifdefs = [true]
			
			# Process the file line by line
			block_comment = false
			File.read(File.join(@directory,filename)).each_line.with_index do |line,index|
				
				# Don't consider one-line comments
				if !(line =~ /^#{WHITESPACE}\/\//)

					# Found the start of a block comment
					if line =~ /\/\*/
						block_comment = true
					end

					# Search for the end of the block comment
					if block_comment
						if line =~ /\*\//
							block_comment = false
						end

					# Not in a block-comment
					else
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
			end
		end
		
	# From this point on are the private methods.
	private
		
		# Method to extract the algorithm details from a marker found in code.
		def marker_to_algorithm(marker)
			algorithm = marker.strip.scan(/^#{WHITESPACE}#{SPECIES_START} (.+)/).join
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

