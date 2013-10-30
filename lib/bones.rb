
# Include the common part between Bones and Aset
require 'common.rb'

# We define a custom error class for code generation related
# errors (any error raised by Bones).
class CodeGenError < StandardError #:nodoc:
end
def raise_error(message) #:nodoc:
	puts Bones::ERROR+message
	raise CodeGenError, 'Error encountered, stopping execution of Bones'
end

# Extending the Ruby standard string class to support some
# additional methods: two methods related to comma removal.
class String #:nodoc:
	
	# Replace double comma's in a string with a single comma.
	# This method is useful for function-argument lists.
	def remove_double_commas
		return self.gsub!(/(,\s*){2,}/,',')
	end
	
	# Remove a comma before a closing bracket in a string, for
	# example: ',)'. This method is useful for function-argument
	# lists.
	def remove_extra_commas
		return self.gsub!(/,\s*\)/,')')
	end
	
	# This function repeatedly applies the remove double commas
	# and remove extra commas functions
	def remove_extras
		return self.remove_double_commas.remove_extra_commas.remove_double_commas.remove_extra_commas
	end
	
end

# The Bones module keeps all the Bones classes and constants
# together. It contains the classes:
# * Engine          The main component of the Bones tool, providing the high-level tool flow.
# * Preprocessor    A C-preprocessor, extracting class information from source code.
# * Algorithm       An individual algorithm, containing an algorithm classification, code and more.
# * Species         A class representing an algorithm class (or: species).
# * Variable        A class for individual variables (not related to CAST).
# * Structure       A class inheriting from the String class, representing parts of the algorithm classification.
# 
# The module also contains a list of inter-class constants.
module Bones
	
	# Set the newline character
	NL = "\n"
	# Set the tab size (currently: 2 spaces)
	INDENT = '  '
	
	# A string given as a start of an informative message. See also ERROR and WARNING.
	MESSAGE = '[Bones] ### Info   : '
	# A string given as a start of an warning message. See also ERROR and MESSAGE.
	WARNING = '[Bones] ### Warning: '
	# A string given as a start of an error message. See also MESSAGE and WARNING.
	ERROR   = '[Bones] ### Error  : '
	
	# Gives a string representing an read-only variable. See also OUTPUT, INOUT and DIRECTIONS.
	INPUT = 'in'
	# Gives a string representing an write-only variable. See also INPUT, INOUT and DIRECTIONS.
	OUTPUT = 'out'
	# Gives a string representing an read/write variable. See also INPUT, OUTPUT and DIRECTIONS.
	INOUT = 'inout'
	# Gives a list of all directions considered. Makes use of the INPUT and OUTPUT constants.
	DIRECTIONS = [INPUT,OUTPUT]
	
	# A string representing the combination character ('^') of a species. See also ARROW and PIPE.
	WEDGE = '^'
	# A string representing the production character ('->') of a species. See also WEDGE and PIPE.
	ARROW = '->'
	# A string representing the pipe character ('|') of a species. See also WEDGE and ARROW.
	PIPE = '|'
	# A string representing the colon character (':') to separate ranges in dimensions.
	RANGE_SEP = ':'
	# A string representing the comma character (',') to separate different ranges.
	DIM_SEP = ','
	
	# Sets the prefix used by variables in the skeleton library. This is used in LOCAL_MEMORY, GLOBAL_ID, LOCAL_ID, GLOBAL_SIZE and LOCAL_SIZE.
	VARIABLE_PREFIX = 'bones_'
	# Sets the variable name for the local memory variable in the skeleton library.
	LOCAL_MEMORY = VARIABLE_PREFIX+'local_memory'
	# Sets the variable name for the thread private (i.e. register) memory variable in the skeleton library.
	PRIVATE_MEMORY = VARIABLE_PREFIX+'private_memory'
	# Sets the variable name for the global memory thread index in the skeleton library.
	GLOBAL_ID = VARIABLE_PREFIX+'global_id'
	# Sets the variable name for the local memory thread index in the skeleton library.
	LOCAL_ID = VARIABLE_PREFIX+'local_id'
	# Sets the variable name for the global memory size as used in the skeleton library.
	GLOBAL_SIZE = VARIABLE_PREFIX+'global_size'
	# Sets the variable name for the local memory size as used in the skeleton library.
	LOCAL_SIZE = VARIABLE_PREFIX+'local_size'
	# Provide a function definition for the initialization C-code (if present). See als INITIALIZATION_CODE.
	INITIALIZATION_DEFINITION = 'void '+VARIABLE_PREFIX+'initialize_target(void);'
	# Provide a function call to the initialization C-code (if present). See als INITIALIZATION_DEFINITION.
	INITIALIZATION_CODE = VARIABLE_PREFIX+'initialize_target();'
	# Sets the name for the 'golden' output, required for verification purposes.
	GOLDEN = VARIABLE_PREFIX+'golden'
	# Sets the loop variable name for the 'golden' output, required for verification purposes.
	LOOP = VARIABLE_PREFIX+'loop'
	# Constant to set the device variable name
	DEVICE = 'device_'
	
	# Provides the starting marker for a search-and-replace variable. See also SAR_MARKER2.
	SAR_MARKER1 = '<'
	# Provides the ending marker for a search-and-replace variable. See also SAR_MARKER1.
	SAR_MARKER2 = '>'
	
	# Set the start of a function definition, used in the skeleton library files. See also END_DEFINITION.
	START_DEFINITION = '\/\* STARTDEF'
	# Set the end of a function definition, used in the skeleton library files. See also START_DEFINITION.
	END_DEFINITION = 'ENDDEF \*\/'
	
	# This class is created to be a parent class of the Bones
	# engine, the Bones species and the Bones algorithm class.
	class Common
	
		# Helper to obtain the 'from' part from a range. For example,
		# the method will yield '1' if applied to '1:N-1'.
		def from(range)
			return '('+simplify(range.split(RANGE_SEP)[0])+')'
		end
	
	
		# Helper to obtain the 'to' part from a range. For example,
		# the method will yield 'N-1' if applied to '1:N-1'.
		def to(range)
			return '('+simplify(range.split(RANGE_SEP)[1])+')'
		end
		
		# Helper to obtain the sum of a range. For example, the method
		# will yield 'N-2' if applied to '1:N-1'. There is a check to
		# ensure that the range is correct.
		def sum(range)
			raise_error('Incorrect range given: "'+range+'"') if range.split(RANGE_SEP).length != 2
			return '('+simplify("(#{to(range)}-#{from(range)}+1)")+')'
		end
		
		
		# Helper to obtain the sum and 'from' part of a range. For
		# example, the method will yield '((N-2)+1)' if applied to '1:N-1'.
		def sum_and_from(range)
			return '('+simplify(sum(range)+'+'+from(range))+')'
		end
		
		# This method flattens a multidimensional hash into a one
		# dimensional hash. This method is called recursively.
		def flatten_hash(hash,flat_hash={},prefix='')
			hash.each do |key,value|
				if value.is_a?(Hash)
					flatten_hash(value,flat_hash,prefix+key.to_s+'_')
				else
					flat_hash[(prefix+key.to_s).to_sym] = value
				end
			end
			return flat_hash
		end
		
		# This method performs a search-and-replace. It searches
		# for the <index> of the input hash and replaces it with
		# the corresponding key. It searches for to-be-replaced
		# variables of the form '<name>'. If such patterns still
		# occur after searching and replacing, the method raises
		# an error.
		def search_and_replace!(hash,code)
			flat_hash = flatten_hash(hash)
			2.times do
				flat_hash.each { |search,replace| code.gsub!(SAR_MARKER1+search.to_s+SAR_MARKER2,replace) }
			end
			raise_error('Unrecognized replace variable "'+($~).to_s+'" in the skeleton library') if code =~ /<[a-zA-Z]+\w*>/
		end
	
		# This method calls search_and_replace! to replaces markers
		# in code with a hash of search-and-replace values. Before,
		# it clones the existing code so that the original copy is
		# maintained.
		def search_and_replace(hash,code)
			new_code = code.clone
			search_and_replace!(hash,new_code)
			return new_code
		end
		
		# Method to process the defines in a piece of code. The code
		# must be formatted as a string. It returns a copy of the code
		# with all defines replaced.
		def replace_defines(original_code,defines)
			code = original_code.clone
			list = defines.sort_by { |key,value| key.to_s.length }
			list.reverse.each do |pair|
				search = pair[0].to_s
				replace = pair[1]
				code.gsub!(search,replace)
			end
			return code
		end
		
	end
	
end

# This list of require's makes sure all Bones classes are
# included. The order is not important here.
require 'bones/structure.rb'
require 'bones/species.rb'
require 'bones/algorithm.rb'
require 'bones/variablelist.rb'
require 'bones/variable.rb'
require 'bones/copy.rb'
require 'bones/preprocessor.rb'
require 'bones/engine.rb'

