
module Adarwin
	
	# This is the C99 pre-processor for Adarwin. It has the following tasks:
	# * Extract the SCoP part from the code (the region of interest)
	# * Extract the header code (defines, includes, etc.)
	# * Output the original code without pre-processor directives
	# * Output the original code minus the SCoP (SCoP to be filled in later)
	class Preprocessor < Common
		attr_reader :source_code, :header_code, :parsed_code, :scop_code, :target_code
		
		# Regular expression to identify whitespaces (tabs, spaces).
		WHITESPACE = '\s*'
		
		# This is the method which initializes the preprocessor. Initialization
		# requires the target source code to process, which is then set as the class
		# variable +@source_code+.
		def initialize(source_code)
			@source_code = source_code
			@header_code = ''
			@parsed_code = ''
			@target_code = ''
			@scop_code = ''
		end
		
		# This is the method to perform the actual preprocessing. This method takes
		# care of all the pre-processor tasks. The output is stored in the two
		# attributes +header_code+, and +scop+.
		# FIXME: What about multi-line statements? For example, a multi-line comment
		# could have a commented-out SCoP or define or include.
		def process
			scop = false
			scop_in_code = false
			
			# Process the file line by line
			@source_code.each_line.with_index do |line,index|
				if line =~ /^#{WHITESPACE}#/
					
					# Keep 'include' statements as header code
					if line =~ /^#{WHITESPACE}#include/
						@header_code += line
						@target_code += line
					
					# Process 'define' statements
					elsif line =~ /^#{WHITESPACE}#define/
						@header_code += line
						@target_code += line
					
					# Found the start of a SCoP
					elsif line =~ /^#{WHITESPACE}#{SCOP_START}/
						scop = true
						scop_in_code = true
						@parsed_code += '{'+NL
						
					# Found the end of a SCoP
					elsif line =~ /^#{WHITESPACE}#{SCOP_END}/
						scop = false
						@parsed_code += '}'+NL
					end
					
				# Nothing special in the code going on here
				else
					@scop_code += line if scop
					@parsed_code += line
					@target_code += line
				end
			end
			
			# Exit if there is no SCoP found
			if !scop_in_code
				raise_error('No "#pragma scop" found in the source code')
			end
		end
	end
end

