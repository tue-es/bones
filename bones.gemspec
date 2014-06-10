Gem::Specification.new do |bones|

	# General
	bones.name        = 'bones-compiler'
	bones.version     = File.read('VERSION').strip
	bones.date        = Date.today.to_s
	bones.license     = 'LICENSE'
	
	# Gem description and documentation
	bones.summary     = "The Bones source-to-source compiler"
	bones.description = "Bones is a source-to-source compiler based on algorithmic skeletons and algorithmic species. It takes C code as input (annotated with species information by A-Darwin), and generates parallel code in languages such as CUDA, OpenCL, and OpenMP. The generated code can be executed on a GPU or a multi-core CPU."
	bones.rdoc_options << '--title' << 'Bones' << '--line-numbers'


	# Author information
	bones.author      = 'Cedric Nugteren'
	bones.email       = 'c.nugteren@tue.nl'
	bones.homepage    = 'http://parse.ele.tue.nl/bones/'

	# Dependencies
	bones.add_dependency 'rake'
	bones.add_dependency 'trollop'
	bones.add_dependency 'cast'
	bones.add_dependency 'symbolic'

	# Executables
	bones.bindir = 'bin'
	bones.executables << 'bones'
	bones.executables << 'adarwin'

	# Files
	bones.extra_rdoc_files = ['README.rdoc']
	bones.files            = Dir['Rakefile', '{bin,examples,lib,skeletons,test}/**/*', 'README.rdoc', 'LICENSE', 'CHANGELOG', 'VERSION']
end
