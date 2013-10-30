
# Include the common part between Bones and A-Darwin
require 'common.rb'

# We define a custom error class for code generation related
# errors (any error raised).
class CodeGenError < StandardError #:nodoc:
end
def raise_error(message) #:nodoc:
	puts Adarwin::ERROR+message
	raise CodeGenError, 'Error encountered, stopping execution of A-Darwin'
end

# The module keeps all the classes and constants
# together. It contains the classes:
# * Engine: The main component of the tool, providing the high-level flow.
# * Preprocessor: C-preprocessor, extracting defines/includes from source code.
# * Nest:
# * Interval:
# * Dependence:
# * Reference:
# 
# The module also contains a list of inter-class constants.
module Adarwin
	
	# A string given as a start of an informative message.
	MESSAGE = '[A-Darwin] ### Info   : '
	# A string given as a start of an warning message.
	WARNING = '[A-Darwin] ### Warning: '
	# A string given as a start of an error message.
	ERROR   = '[A-Darwin] ### Error  : '
	
	# Start of the scop
	SCOP_START = '#pragma scop'
	# Enf of the scop
	SCOP_END = '#pragma endscop'
	
	# Species pragma
	PRAGMA_SPECIES = '#pragma species'
	
	# Array reference characterisation (ARC) pragma
	PRAGMA_ARC = '#pragma ARC'
	
	# Create a string from a pragma because pragma's are unsupported by CAST.
	PRAGMA_DELIMITER_START = '"PRAGMA '
	PRAGMA_DELIMITER_END = ' PRAGMA"'
	
	# This class is created to be a parent class of all classes.
	class Common
	end
	
end

# This list of require's makes sure all classes are included.
require 'adarwin/interval.rb'
require 'adarwin/dependences.rb'
require 'adarwin/preprocessor.rb'
require 'adarwin/memorycopies.rb'
require 'adarwin/fusion.rb'
require 'adarwin/engine.rb'
require 'adarwin/reference.rb'
require 'adarwin/nest.rb'