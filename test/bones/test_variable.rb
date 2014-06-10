# Include the test helper
require File.dirname(__FILE__) + '/../test_helper'

# Test class for the variable class
class TestVariable < Test::Unit::TestCase
	
	# Some constants to test against.
	NAME = 'example'
	
	# Method to create variable examples from code examples.
	def setup
		@variables = []
		@dimensions = []
		parser = C::Parser.new
		prefix = 'void main() {'
		suffix = '}'
		
		# Create code examples
		code_examples = []
		@types = []
		typeprefixes = ['int','float','int *','int **','int ***','unsigned char *']
		typesuffixes = ['','[10]','[N]','[10][10]']
		typeprefixes.each do |typeprefix|
			typesuffixes.each do |typesuffix|
				@types.push([typeprefix,typesuffix])
				@types.push([typeprefix,typesuffix])
			end
		end
		@types.each_with_index do |type,index|
			if index.odd?
				definition = type[0]+' '+NAME+type[1]
				code_examples.push(parser.parse([prefix,definition+' = 3;',suffix].join("\n")))
				code_examples.push(parser.parse([prefix,definition+';','int a = '+NAME+';',suffix].join("\n")))
			end
			@dimensions.push(type[0].scan('*').length + type[1].scan('[').length)
		end
		
		# Create variables
		code_examples.each do |code|
			@variables.push(Bones::Variable.new(NAME,code.variable_type(NAME),code.size(NAME),Bones::INPUT,'0',false)) 
		end
	end
	
	# Test whether the typename of the variable is recognized correctly.
	def test_typename
		@variables.each_index do |index|
			assert_equal(@types[index][0].gsub('*','').strip,@variables[index].type_name)
		end
	end
	
	# Test whether the device pointer is obtained correctly.
	def test_device_pointer
		@variables.each_index do |index|
			expected_result = (@dimensions[index] == 0) ? '' : '*'
			assert_equal(expected_result,@variables[index].device_pointer)
		end
	end
	
	# Test whether the dimension of a variable is obtained correctly.
	def test_dimension
		@variables.each_index do |index|
			assert_equal(@dimensions[index],@variables[index].dimensions)
		end
	end
	
	# Test to see if the flattened array is obtained correctly.
	def test_flatten
		@variables.each_index do |index|
			if @variables[index].dimensions > 1
				expected_result = ''+'[0]'*(@dimensions[index]-1)
				assert_equal(expected_result,@variables[index].flatten)
			end
		end
	end
	
	# Test whether the variable definition is obtained correctly.
	def test_definition
		@variables.each_index do |index|
			expected_result = @types[index][0]+' '+NAME+@types[index][1]
			assert_equal(expected_result,@variables[index].definition)
		end
	end
	
end
