# Include the test helper
require File.dirname(__FILE__) + '/../test_helper'

# Test class for the species class
class TestSpecies < Test::Unit::TestCase
	
	# Create a comprehensive list of known species.
	def setup
		list = setup_species
		@dimensions = list[:dimensions]
		@inputs = list[:inputs]
		@outputs = list[:outputs]
		@patterns = list[:patterns]
		@prefixes = list[:prefixes]
		@species = list[:species]
	end
	
	def test_nothing
	end
	
end
