require 'rake/testtask'
require 'rdoc/task'
require 'rake/clean'

# Set the location of the examples
EXAMPLES = File.join('examples','benchmarks','*.c')

# Set the clean/clobber tasks
CLOBBER.include(Dir[
	File.join('examples','*_species.c'),
	File.join('examples','*','*_*-*'),
	File.join('examples','*','*_species.c'),
	File.join('examples','benchmarks','*','*_*-*'),
	File.join('examples','benchmarks','*','*_species.c')
])

# Set the location of the examples
ADARWIN_EXAMPLES_ALL = [
	File.join('examples','element','*.c'),
	File.join('examples','chunk','*.c'),
	File.join('examples','neighbourhood','*.c'),
	File.join('examples','shared','*.c'),
	File.join('examples','dependences','*.c'),
	File.join('examples','benchmarks','PolyBench','*.c')
]
# Select PolyBench as the set of examples
ADARWIN_EXAMPLES = ADARWIN_EXAMPLES_ALL[5]
ADARWIN_MEMORY = false unless defined?(ADARWIN_MEMORY)
ADARWIN_OPTIONS = ADARWIN_MEMORY ? '-r -f -b -l' : '--no-memory-annotations' unless defined?(ADARWIN_OPTIONS)

# Pick a target from a list of possible targets
#               0             1                2                  3              4         5   
TARGETS = ['GPU-CUDA','GPU-OPENCL-AMD','CPU-OPENCL-INTEL','CPU-OPENCL-AMD','CPU-OPENMP','CPU-C']
TARGET = TARGETS[0]

# Settings for Bones
MEASUREMENTS = true
VERIFICATION = false
MEMORY_OPTIMISATIONS = true
ADARWIN_OPTIONS_BONES = MEMORY_OPTIMISATIONS ? '-r -f -b -l' : ''

# Small helper function to display text on screen
def display(text)
	print '[Rake] ### '+text+': '
	p
end

# Set the default task
task :default => [:examples]

# Rake tasks related to the examples
namespace :examples do
	
	# Task to process and test everything (generating code, compiling code, executing)
	desc 'Run all examples through Bones, compile them, and execute them'
	task :verify, [:file] => [:generate, :compile, :execute] do |t, args|
	end
	
	# Task to process and test everything through A-Darwin and Bones (generating code, compiling code, executing)
	desc 'Run the examples through A-Darwin and Bones, then compile and execute them'
	task :go, :file do |t, args|
		bones_options = (MEASUREMENTS ? '-m ' : '') + (VERIFICATION ? '-c ' : '')
		args.with_defaults(:file => EXAMPLES)
		Dir[args.file].sort.each do |file|
			sh "bin/adarwin -a #{file} #{ADARWIN_OPTIONS_BONES}"
			split = file.split('.')
			file = split[0]+'_species'+'.'+split[1]
			sh "bin/bones -a #{file} -t #{TARGET} #{bones_options}"
			compile(file,TARGET)
			execute(file,TARGET)
		end
	end
	
	# Task to pass examples through Bones
	desc 'Generate target code using Bones'
	task :generate, :file do |t, args|
		options = (MEASUREMENTS ? '-m ' : '') + (VERIFICATION ? '-c ' : '')
		args.with_defaults(:file => EXAMPLES)
		Dir[args.file].sort.each do |file|
			display('Generating')
			sh "bin/bones -a #{file} -t #{TARGET} #{options}"
		end
	end
	
	# Task to compile the generated code for the examples (NOTE: this task is a stub)
	desc 'Compile all examples (using gcc/nvcc)'
	task :compile, :file do |t, args|
		args.with_defaults(:file => EXAMPLES)
		Dir[args.file].sort.each do |file|
			compile(file,TARGET)
		end
	end
	
	# Task to execute the compiled code for the examples (NOTE: this task is a stub)
	desc 'Execute all examples'
	task :execute, :file do |t, args|
		args.with_defaults(:file => EXAMPLES)
		Dir[args.file].sort.each do |file|
			execute(file,TARGET)
		end
	end

	# Helper function to compile code (NOTE: this task is a stub)
	def compile(file,target)
		puts "[Rake] ### Compiling the code is system-specific, to be filled in..."
	end

	# Helper function to execute code (NOTE: this task is a stub)
	def execute(file,target)
		puts "[Rake] ### Executing the code is system-specific, to be filled in..."
	end
	
end
task :examples => ['examples:generate']

# Task which adds a new target to the skeleton library based on an existing target
desc 'Adds a new target to the skeleton library'
task :add_target, :name, :base do |t, args|
	args.with_defaults(:name => 'NEW-TARGET', :base => 'CPU-OPENMP')
	base = 'skeletons/'+args.base
	name = 'skeletons/'+args.name
	if File.exists?(base) && !File.exists?(name)
		sh "cp -r #{base} #{name}"
	else
		puts '[Rake] ### Error adding new target'
	end
end

# Generate species descriptions using A-Darwin
desc 'Extract species descriptions using A-Darwin'
task :adarwin, :file do |t, args|
	args.with_defaults(:file => ADARWIN_EXAMPLES)
	Dir[args.file].sort.each do |file|
		adarwin(file,ADARWIN_OPTIONS)
	end
end

# Generate species descriptions using A-Darwin
desc 'Test A-Darwin`s output against golden samples'
task :adarwin_test do |t|
	pass = 0
	fail = 0
	ADARWIN_EXAMPLES_ALL.each do |examples|
		Dir[examples].sort.each do |file|
			if !(file =~ /_species\.c/)
				adarwin(file,'--no-memory-annotations')
				display('Testing correctness')
				speciesfile = file.gsub('.c','_species.c')
				sh "diff #{speciesfile} test/#{speciesfile}" do |ok,status|
					ok ? pass += 1 : fail += 1
				end
			end
		end
	end
	display('Test results')
	puts "PASS: #{pass}, FAIL: #{fail}"
end

# Method to run A-Darwin for a set of files
def adarwin(file,options)
	if !(file =~ /_species\.c/)
		display('Extracting species')
		sh "bin/adarwin --application #{file} --silent #{options}"
	end
end

# Test individual parts of the code
Rake::TestTask.new do |test|
	test.test_files = FileList[File.join('test','*','test_*.rb')]
	test.verbose = false
end

# Generate HTML documentation using RDoc
RDoc::Task.new do |rdoc|
	rdoc.title = 'Bones'
	rdoc.options << '--line-numbers'
	rdoc.rdoc_files.include(File.join('lib','**','*.rb'))
	rdoc.rdoc_files.include('README.rdoc')
	rdoc.rdoc_dir = 'rdoc'
	rdoc.main = 'README.rdoc'
end

