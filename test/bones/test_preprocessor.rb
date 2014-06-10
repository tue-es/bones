# Include the test helper
require File.dirname(__FILE__) + '/../test_helper'

# Test class for the preprocessor class.
class TestPreprocessor < Test::Unit::TestCase
	
	# Create a list of known examples and reference results.
	def setup
		
		# Create a comprehensive list of known speciess
		list = setup_species
		@examples = list[:examples]
		
		# Create a list of corresponding algorithms and code
		@algorithms_list, code_list = setup_algorithms(@examples)
		
		# Create and execute the preprocessors
		@preprocessors = []
		code_list.each_index do |index|
			preprocessor = Bones::Preprocessor.new(code_list[index],'','')
			preprocessor.process
			@preprocessors.push(preprocessor)
		end
	end
	
	# Method to test the found algorithms (species part).
	def test_algorithms_species
		@preprocessors.each_with_index do |preprocessor,index1|
			reference_algorithms = @algorithms_list[index1]
			preprocessor.algorithms.each_with_index do |algorithm,index2|
				assert_equal(reference_algorithms[index2].species.prefix,algorithm.species.prefix)
			end
		end
	end
	
	# Method to test the found algorithms (code part).
	def test_algorithms_code
		@preprocessors.each_with_index do |preprocessor,index1|
			reference_algorithms = @algorithms_list[index1]
			preprocessor.algorithms.each_with_index do |algorithm,index2|
				assert_equal(reference_algorithms[index2].code,algorithm.code)
			end
		end
	end
end

