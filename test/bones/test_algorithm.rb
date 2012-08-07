# Include the test helper
require File.dirname(__FILE__) + '/../test_helper'

# Test class for the primitive class.
class TestAlgorithm < Test::Unit::TestCase
	
	# Create a list of known examples and the results.
	def setup
		
		# Create a comprehensive list of known tribes
		list = setup_species
		@examples = list[:examples]
		@defines = []
		
		# Create a list of corresponding preprocessors and code
		@primitives_list, original_code_list, @arrays_list = setup_algorithms(@examples)
		
		# Use the preprocessor and the 'CAST' gem to create an AST of the original code
		original_ast_list = []
		original_code_list.each do |original_code|
			preprocessor = Bones::Preprocessor.new(original_code,'','')
			preprocessor.process
			@defines.push(preprocessor.defines)
			original_ast_list.push(C.parse(preprocessor.target_code))
		end
		
		# Populate the contents of the primitives
		@primitives_list.each_with_index do |primitives,index|
			primitives.each do |algorithm|
				algorithm.populate_lists()
				#algorithm.populate_hash()
			end
		end
	end
	
	def test_nothing
	end
	
end

