require 'rake/testtask'
require 'rdoc/task'
require 'rake/clean'

# Set the location of the examples
EXAMPLES = File.join('examples','benchmarks','*.c')

# Set the clean/clobber tasks
CLOBBER.include(Dir[File.join('examples','*','*_*-*')])

# Pick a target from a list of possible targets
#               0             1                2                  3              4         5   
TARGETS = ['GPU-CUDA','GPU-OPENCL-AMD','CPU-OPENCL-INTEL','CPU-OPENCL-AMD','CPU-OPENMP','CPU-C']
TARGET = TARGETS[0]

# Settings for Bones
MEASUREMENTS = true
VERIFICATION = true

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
	
	# Task to pass examples through Bones
	desc 'Generate target code using Bones'
	task :generate, :file do |t, args|
		args.with_defaults(:file => EXAMPLES)
		Dir[args.file].sort.each do |file|
			display('Generating')
			options = (MEASUREMENTS ? '-m ' : '') + (VERIFICATION ? '-c ' : '')
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

	# Helper function to compile code
	#def compile(file,target)
		# (system-specific, to be filled in by the user)
	#end

	# Helper function to execute code
	#def execute(file,target)
		# (system-specific, to be filled in by the user)
	#end
	
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

