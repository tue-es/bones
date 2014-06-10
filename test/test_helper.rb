
# Load the test from the Ruby standard library
require 'test/unit'

# Set the path for the libraries and the installation directory
BONES_DIR = File.dirname(__FILE__) + '/../'
lib_dir = BONES_DIR+'lib'
$LOAD_PATH.unshift lib_dir unless $LOAD_PATH.include?(lib_dir)

# Set verbose output to false
VERBOSE = false

# Set command line arguments
ARGUMENTS = ['-a','test.c','-t','gpu-nvidia']

# Load the test file and the bones library
require 'castaddon'
require 'bones'

# Set some constants to test with
MAX_DIMENSIONS = 3
MAX_INPUTS = 2
MAX_OUTPUTS = 1
SIZE = 30
PATTERNS = ['element','neighbourhood','chunk','shared']
PREFIXES = ['','unordered']
PIPE = '|'

# Create a list of speciess to test with.
def setup_species
	list = {:dimensions => [], :inputs => [], :outputs => [], :patterns => [], :prefixes => [], :species => [], :examples => []}
	for num_outputs in 1..MAX_OUTPUTS
		for num_inputs in 1..MAX_INPUTS
			for num_dimensions in 1..MAX_DIMENSIONS
				PREFIXES.each do |prefix|
					PATTERNS.each do |pattern|
						
						# Create the species information
						range = '0'+Bones::RANGE_SEP+(SIZE-1).to_s
						dimension = range+(Bones::DIM_SEP+range)*(num_dimensions-1)
						pattern_parameter = (pattern == 'neighbourhood' || pattern == 'chunk') ? pattern+'('+dimension+')' : pattern
						inputs = dimension+PIPE+pattern_parameter+(' '+Bones::WEDGE+' '+dimension+PIPE+pattern_parameter)*(num_inputs-1)
						outputs = dimension+PIPE+pattern_parameter+(' '+Bones::WEDGE+' '+dimension+PIPE+pattern_parameter)*(num_outputs-1)
						
						# Set the species information
						list[:dimensions].push(num_dimensions)
						list[:inputs].push(num_inputs)
						list[:outputs].push(num_outputs)
						list[:patterns].push(pattern)
						list[:prefixes].push(prefix)
						list[:species].push(Bones::Species.new(prefix,inputs,outputs))
						list[:examples].push(prefix+' '+inputs+' '+Bones::ARROW+' '+outputs)
					end
				end
			end
		end
	end
	return list
end

# Construct a list of primitives corresponding to the example
# speciess created in the function 'setup speciess'. This function
# also creates code examples.
def setup_algorithms(examples)
	primitives_list = []
	original_code_list = []
	arrays_list = []
	code_prefix = "void main() { \n float in[10][10]; \n"
	primitive_code_start = "{\n int i = 4; \n"
	primitive_code_end = "}"
	code_suffix = "\n }"

	# Iterate over the example speciess
	examples.each do |example|
		
		# Create an example primitive
		primitive_name = 'example'
		primitive_start = Bones::Preprocessor::PRIMITIVE_START+' '+example
		primitive_end = Bones::Preprocessor::PRIMITIVE_END+' '+primitive_name

		# Create a matching species for comparison
		prefix = ''
		if example =~ /^(unordered|multiple)/
			partition = example.partition(' ')
			example = partition[2]
			prefix = partition[0]
		end
		example_split = example.split(' '+Bones::ARROW+' ')
		species = Bones::Species.new(prefix,example_split[0].strip,example_split[1].strip)
		
		# Set the code according to the primitive
		species.outputs.length.times { |i| code_prefix += "int * out_"+i.to_s+" = malloc("+SIZE.to_s+"*sizeof(int)); \n" }
		species.inputs.length.times { |i| code_prefix += "float in_"+i.to_s+"[10][10]; \n" }
		primitive_code = primitive_code_start
		species.inputs.length.times { |i| primitive_code += "float a; \n a = in_"+i.to_s+"[6][5]; \n" }
		species.outputs.length.times { |i| primitive_code += "out_"+i.to_s+"[i] = a * 2; \n" }
		primitive_code += primitive_code_end
		
		# Create corresponding original code
		original_code_list.push(code_prefix+"\n"+primitive_start+"\n"+primitive_code+"\n"+primitive_end+"\n"+code_suffix)
		primitives_list.push([Bones::Algorithm.new(primitive_name,primitive_name,'0',species,primitive_code)])
		arrays_list.push(['in','out'])
	end
	return primitives_list, original_code_list, arrays_list
end

