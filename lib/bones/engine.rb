
module Bones
	# This class holds the main functionality: the Bones source-
	# to-source compilation engine based on algorithmic skeletons.
	# This class processes command line arguments, makes calls to
	# the Bones preprocessor and the CAST gem, analyzes the source
	# code, performs source transformations, instantiates the 
	# skeletons, and finally writes output code to file.
	class Engine < Common
		
		# Locate the skeletons directory.
		BONES_DIR_SKELETONS = File.join(BONES_DIR,'skeletons')
		
		# Set the name of the transformations file as found in the skeleton library.
		SKELETON_FILE = 'skeletons.txt'
		
		# A list of timer files to be found in the skeleton library.
		TIMER_FILES = ['timer_1_start','timer_1_stop','timer_2_start','timer_2_stop']
		# A list of files to be found in the common directory of the skeleton library (excluding timer files).
		COMMON_FILES = ['prologue','epilogue','mem_prologue','mem_copy_H2D','mem_copy_D2H','mem_epilogue','mem_global']
		# The name of the file containing the globals as found in the skeleton library
		COMMON_GLOBALS = 'globals'
		# The name of the file containing the header file for the original C code as found in the skeleton library
		COMMON_HEADER = 'header'
		# The name of the file containing the globals for the kernel files as found in the skeleton library
		COMMON_GLOBALS_KERNEL = 'globals_kernel'
		# The name of the file containing the scheduler code
		COMMON_SCHEDULER = 'scheduler'
		# Global timers
		GLOBAL_TIMERS = 'timer_globals'
		
		# The extension of a host file in the skeleton library. See also SKELETON_DEVICE.
		SKELETON_HOST = '.host'
		# The extension of a device file in the skeleton library. See also SKELETON_HOST.
		SKELETON_DEVICE = '.kernel'
		
		# The suffix added to the generated output file for the host file. See also OUTPUT_DEVICE.
		OUTPUT_HOST = '_host'
		# The suffix added to the generated output file for the device file. See also OUTPUT_HOST.
		OUTPUT_DEVICE = '_device'
		# The suffix added to the generated verification file. See also OUTPUT_DEVICE and OUTPUT_HOST.
		OUTPUT_VERIFICATION = '_verification'
		
		# Initializes the engine and processes the command line
		# arguments. This method uses the 'trollop' gem to parse
		# the arguments and to create a nicely formatted help menu.
		# This method additionally initializes a result-hash and
		# reads the contents of the source file from disk.
		#
		# ==== Command-line usage:
		#    bones --application <input> --target <target> [OPTIONS]
		#
		# ==== Options:
		#  --application, -a <s>:   Input application file
		#       --target, -t <s>:   Target processor (choose from: 'GPU-CUDA','GPU-OPENCL-AMD','CPU-OPENCL-INTEL','CPU-OPENCL-AMD','CPU-OPENMP','CPU-C')
		#     --measurements, -m:   Enable/disable timers
		#          --version, -v:   Print version and exit
		#             --help, -h:   Show this message
		#
		def initialize
			@result = {:original_code            => [],
			           :header_code              => [],
			           :host_declarations        => [],
			           :host_code_lists          => [],
			           :algorithm_declarations   => [],
			           :algorithm_code_lists     => [],
			           :verify_code              => [],
			           :host_device_mem_globals  => []}
			@state = 0
			
			# Provides a list of possible targets (e.g. GPU-CUDA, 'CPU-OPENCL-INTEL').
			targets = []
			Dir[File.join(BONES_DIR_SKELETONS,'*')].each do |entry|
				if (File.directory?(entry)) && !(entry =~ /verification/)
					targets.push(File.basename(entry))
				end
			end
			targets = targets.sort
			
			# Parse the command line options using the 'trollop' gem.
			pp_targets = targets.inspect.gsub(/("|\[)|\]/,'')
			@options = Trollop::options do
				version 'Bones '+File.read(BONES_DIR+'/VERSION').strip+' (c) 2012 Cedric Nugteren, Eindhoven University of Technology'
				banner  NL+'Bones is a parallelizing source-to-source compiler based on algorithmic skeletons. ' +
				        'For more information, see the README.rdoc file or visit the Bones website at http://parse.ele.tue.nl/bones/.' + NL + NL +
				        'Usage:' + NL +
				        '    bones --application <input> --target <target> [OPTIONS]' + NL +
				        'using the following flags:'
				opt :application,     'Input application file',                               :short => 'a', :type => String
				opt :target,          'Target processor (choose from: '+pp_targets+')',       :short => 't', :type => String
				opt :measurements,    'Enable/disable timers',                                :short => 'm', :default => false
				opt :verify,          'Verify correctness of the generated code',             :short => 'c', :default => false
				opt :only_alg_number, 'Only generate code for the x-th species (99 -> all)',  :short => 'o', :type => Integer, :default => 99
				opt :merge_factor,    'Thread merge factor, default is 1 (==disabled)',       :short => 'f', :type => Integer, :default => 0
				opt :register_caching,'Enable register caching: 1:enabled (default), 0:disabled',      :short => 'r', :type => Integer, :default => 1
				opt :zero_copy       ,'Enable OpenCL zero-copy: 1:enabled (default), 0:disabled',      :short => 'z', :type => Integer, :default => 1
				opt :skeletons       ,'Enable non-default skeletons: 1:enabled (default), 0:disabled', :short => 's', :type => Integer, :default => 1
			end
			Trollop::die 'no input file supplied (use: --application)'              if !@options[:application_given]
			Trollop::die 'no target supplied (use: --target)'                       if !@options[:target_given]
			Trollop::die 'input file "'+@options[:application]+'" does not exist'   if !File.exists?(@options[:application])
			Trollop::die 'target not supported, supported targets are: '+pp_targets if !targets.include?(@options[:target].upcase)
			@options[:name] = File.basename(@options[:application], ".*")
			@options[:target] = @options[:target].upcase
			
			# Extension for the host files corresponding to the target.
			@extension = File.extname(Dir[File.join(BONES_DIR_SKELETONS,@options[:target],'common','*')][0])
			
			# Extension for the device files corresponding to the target.
			@algorithm_extension = File.extname(Dir[File.join(BONES_DIR_SKELETONS,@options[:target],'kernel','*.kernel.*')][0])
			
			# Set a prefix for functions called from the original file but defined in a host file
			@prefix = (@options[:target] == 'GPU-CUDA') ? '' : ''
			
			# Setting to include the scheduler (CUDA only)
			@scheduler = (@options[:target] == 'GPU-CUDA') ? true : false
			
			# Skip analyse passes for certain targets
			@skiptarget = false #(@options[:target] == 'PAR4ALL') ? true : false
			
			# Set the location for the skeleton library
			@dir = {}
			@dir[:library] = File.join(BONES_DIR_SKELETONS,@options[:target])
			@dir[:skeleton_library] = File.join(@dir[:library],'kernel')
			@dir[:common_library] = File.join(@dir[:library],'common')
			@dir[:verify_library] = File.join(BONES_DIR_SKELETONS,'verification')
			
			# Obtain the source code from file
			@source = File.open(@options[:application],'r'){|f| f.read}
			@basename = File.basename(@options[:application],'.c')
		end
		
		# Method to process a file and to output target code. This
		# method calls all relevant private methods.
		#
		# ==== Tasks:
		# * Run the preprocessor to obtain algorithm information.
		# * Use the 'CAST' gem to parse the source into an AST.
		# * Call the code generator to perform the real work and produce output.
		def process
			
			# Run the preprocessor
			preprocessor = Bones::Preprocessor.new(@source,File.dirname(@options[:application]),@basename,@scheduler)
			preprocessor.process
			@result[:header_code] = preprocessor.header_code
			@result[:device_header] = preprocessor.device_header
			@result[:header_code] += '#include <sys/time.h>'+NL if @options[:measurements]
			
			# Parse the source code into AST
			parser = C::Parser.new
			parser.type_names << 'FILE'
			parser.type_names << 'size_t'
			ast = parser.parse(preprocessor.target_code)
			ast.preprocess

			# Add the scheduler's global code
			if @scheduler
				@result[:host_code_lists].push(File.read(File.join(@dir[:common_library],COMMON_SCHEDULER+@extension)))
			end
			
			# Set the algorithm's skeleton and generate the global code
			one_time = true
			preprocessor.algorithms.each_with_index do |algorithm,algorithm_number|
				algorithm.species.set_skeleton(File.join(@dir[:library],SKELETON_FILE))
				if @options[:skeletons] == 0
					algorithm.species.skeleton_name = 'default'
					algorithm.species.settings.gsub!('10','00').gsub!('20','00').gsub!('30','00')
				end
				if algorithm.species.skeleton_name && one_time
					@result[:host_code_lists].push(File.read(File.join(@dir[:common_library],COMMON_GLOBALS+@extension)))
					@result[:algorithm_code_lists].push(File.read(File.join(@dir[:common_library],COMMON_GLOBALS_KERNEL+@extension)))
					one_time = false
				end
			end
			
			# Perform code generation (per-species code)
			@result[:original_code] = ast
			arrays = []
			preprocessor.algorithms.each_with_index do |algorithm,algorithm_number|
				if @options[:only_alg_number] == 99 || algorithm_number == [@options[:only_alg_number],preprocessor.algorithms.length-1].min
					puts MESSAGE+'Starting code generation for algorithm "'+algorithm.name+'"'
					if algorithm.species.skeleton_name
						algorithm.merge_factor = @options[:merge_factor] if (@options[:target] == 'GPU-CUDA')
						algorithm.register_caching_enabled = @options[:register_caching]
						algorithm.set_function(ast)
						algorithm.populate_variables(ast,preprocessor.defines) if !@skiptarget
						algorithm.populate_lists()
						algorithm.populate_hash() if !@skiptarget
						generate(algorithm)
						puts MESSAGE+'Code generated using the "'+algorithm.species.skeleton_name+'" skeleton'
						arrays.concat(algorithm.arrays)
					else
						puts WARNING+'Skeleton "'+algorithm.species.name+'" not available'
					end
				end
			end
			
			# Only if the scheduler is included
			if @scheduler
			
				# Perform code generation (sync statements)
				@result[:host_declarations].push('void bones_synchronize(int bones_task_id);')
				
				# Perform code generation (memory allocs)
				allocs = []
				preprocessor.copies.each do |copy|
					if !allocs.include?(copy.name)
						generate_memory('alloc',copy,arrays,0)
						allocs << copy.name
					end
				end
				
				# Perform code generation (memory copies)
				preprocessor.copies.each_with_index do |copy,index|
					#puts MESSAGE+'Generating copy code for array "'+copy.name+'"'
					generate_memory('copy',copy,arrays,index)
				end
				
				# Perform code generation (memory frees)
				frees = []
				preprocessor.copies.each do |copy|
					if !frees.include?(copy.name)
						generate_memory('free',copy,arrays,0)
						frees << copy.name
					end
				end
			
			end
			
		end

		# This method writes the output code to files. It creates
		# a new directory formatted as 'name_target' and produces
		# three files.
		#
		# ==== Output files:
		# * +main+ -   a file containing the original code with function calls substituting the original algorithms.
		# * +target+ - a file containing the host code for the target.
		# * +kernel+ - a file containing the kernel code for the target.
		def write_output
			
			# Create a new directory for the output
			directory = @options[:application].rpartition('.').first+'_'+@options[:target]
			Dir.mkdir(directory,0744) unless File.directory?(directory)
			
			parser = C::Parser.new
			parser.type_names << 'FILE'
			parser.type_names << 'size_t'
			
			# Populate the main file
			File.open(File.join(directory,@options[:application].split(File::SEPARATOR).last),'w') do |main|
				main.puts '#include <string.h>' if @options[:verify]
				main.puts @result[:header_code]
				main.puts File.read(File.join(@dir[:common_library],COMMON_HEADER+@extension))
				main.puts @result[:host_declarations]
				main.puts
				begin
					main.puts parser.parse(@result[:original_code]).to_s
				rescue
					puts WARNING+'Recovering from CAST parse error'
					main.puts parser.parse(@result[:original_code].clone).to_s
				end
			end
			
			# Populate the verification file
			if @options[:verify]
				File.open(File.join(directory,@options[:name]+OUTPUT_VERIFICATION+@extension),'w') do |verification|
					verification.puts @result[:header_code]
					verification.puts File.read(File.join(@dir[:verify_library],'header.c'))
					verification.puts
					verification.puts @result[:verify_code]
				end
			end
			
			# Populate the target file (host)
			File.open(File.join(directory,@options[:name]+OUTPUT_HOST+@extension),'w') do |target|
				target.puts '#include <cuda_runtime.h>'+NL if @options[:target] == 'GPU-CUDA'
				target.puts "#define ZEROCOPY 0"+NL if @options[:zero_copy] == 0 && @options[:target] == 'CPU-OPENCL-INTEL'
				target.puts "#define ZEROCOPY 1"+NL if @options[:zero_copy] == 1 && @options[:target] == 'CPU-OPENCL-INTEL'
				target.puts @result[:header_code]
				target.puts
				target.puts @result[:host_device_mem_globals]
				target.puts
				target.puts @result[:algorithm_declarations]
				target.puts @result[:host_code_lists]
				target.puts
				target.puts File.read(File.join(@dir[:common_library],GLOBAL_TIMERS+@extension))
			end
			
			# Populate the algorithm file (device)
			File.open(File.join(directory,@options[:name]+OUTPUT_DEVICE+@algorithm_extension),'w') do |algorithm|
				algorithm.puts @result[:device_header]
				algorithm.puts @result[:algorithm_code_lists]
			end
			
		end
		
	# Start of the class's private methods.
	private
		
		# This method takes as an input an indivual algorithm and
		# generates the corresponding output code. The method first
		# creates a search-and-replace hash, after which it instan-
		# tiates a skeleton.
		#
		# This method returns a message informing the user whether
		# the code was succesfully generated or the skeleton was
		# not available.
		def generate(algorithm)
			
			# Determine the skeleton filenames and load them skeletons from the skeleton library
			file_name_host = File.join(@dir[:skeleton_library],algorithm.species.skeleton_name+SKELETON_HOST)
			file_name_device = File.join(@dir[:skeleton_library],algorithm.species.skeleton_name+SKELETON_DEVICE)
			if !File.exists?(file_name_host+@extension) || !File.exists?(file_name_device+@algorithm_extension)
				raise_error('Skeleton files for skeleton "'+algorithm.species.skeleton_name+'" not available')
			end
			skeletons = {:host   => File.read(file_name_host+@extension),
			             :device => File.read(file_name_device+@algorithm_extension)}
			
			# Perform the transformations on the algorithm's code
			algorithm.perform_transformations(algorithm.species.settings) if !@skiptarget
			
			# Load the common skeletons from the skeleton library
			COMMON_FILES.each do |skeleton|
				skeletons[skeleton.to_sym] = File.read(File.join(@dir[:common_library],skeleton+@extension))
			end
			
			# Load the timer code from the skeleton library (only if the '--measurements' flag is given)
			TIMER_FILES.each do |skeleton|
				skeletons[skeleton.to_sym] = @options[:measurements] ? File.read(File.join(@dir[:common_library],skeleton+@extension)) : ''
			end
			
			# Perform search-and-replace on the device skeleton
			search_and_replace!(algorithm.hash,skeletons[:device])
			skeletons[:device].remove_extras
			
			# Replace mathematical functions with their equivalent device functions
			if @options[:target] == 'GPU-CUDA'
				math_functions = {:sqrt => 'sqrtf', :max  => 'fmaxf', :min  => 'fminf'}
				math_functions.each do |original, replacement|
					skeletons[:device].gsub!(/\b#{original}\(/,replacement+'(')
				end
			end
			
			# Create the algorithm declaration list from the header supplied in the skeletons
			algorithm_declaration = skeletons[:device].scan(/#{START_DEFINITION}(.+)#{END_DEFINITION}/m).join.strip.remove_extras
			@result[:algorithm_declarations].push(algorithm_declaration)
			
			# Remove the (commented) algorithm declaration from the code and push the skeleton to the output
			@result[:algorithm_code_lists].push(skeletons[:device].gsub!(/#{START_DEFINITION}(.+)#{END_DEFINITION}/m,''))
			
			# Setup some variables to create the host body function including memory allocation and memory copies
			processed = {:mem_prologue => '', :mem_copy_H2D => '', :mem_copy_D2H => '', :mem_epilogue => ''}
			counter = {:out => 0, :in => 0}
			
			# Iterate over all the array variables and create a mini-search-and-replace hash for each array (all arrays)
			algorithm.arrays.each_with_index do |array, arrayid|
				minihash = { :array               => array.name,
				             :type                => array.type_name,
				             :flatten             => array.flatten,
				             :variable_dimensions => array.size.join('*'),
				             :state               => @state.to_s}
				@state += 1
				
				# Apply the mini-search-and-replace hash to create the memory allocations, memory copies (if input only), etc.
				processed[:mem_prologue] += search_and_replace(minihash,skeletons[:mem_prologue])
				processed[:mem_copy_H2D] += search_and_replace(minihash,skeletons[:mem_copy_H2D]) if array.input? || array.species.shared?
				processed[:mem_epilogue] += search_and_replace(minihash,skeletons[:mem_epilogue])
			
				# Add the device declarations
				@result[:host_device_mem_globals].push(search_and_replace(minihash,skeletons[:mem_global]))
			end
			
			# Iterate over all the array variables and create a mini-search-and-replace hash for each array (output arrays)
			algorithm.arrays.select(OUTPUT).each_with_index do |array, num_array|
				hash = algorithm.hash["out#{num_array}".to_sym]
				minihash = { :array               => array.name,
				             :type                => array.type_name,
				             :flatten             => array.flatten,
				             :offset              => '('+hash[:dimension0][:from]+')',
				             :variable_dimensions => '('+hash[:dimensions]+')',
				             :state               => @state.to_s}
				@state += 1
				
				# Perform selective copy for arrays with 2 dimensions (uses a for-loop over the memory copies)
				if array.dimensions == 2 && @options[:target] == 'GPU-CUDA' && false
					x_from = '('+hash[:dimension0][:from]+')'
					x_to   = '('+hash[:dimension0][:to]+')'
					x_sum  = '('+hash[:dimension0][:sum]+')'
					x_size = array.size[0]
					y_from = '('+hash[:dimension1][:from]+')'
					y_to   = '('+hash[:dimension1][:to]+')'
					y_sum  = '('+hash[:dimension1][:sum]+')'
					y_size = array.size[1]
					processed[:mem_copy_D2H] += NL+INDENT+"for(int bones_x=#{x_from}; bones_x<=#{x_to}; bones_x++) {"+INDENT*2
					minihash[:offset] = "(bones_x*#{y_size})+#{y_from}"
					minihash[:variable_dimensions] = "#{y_sum}"
				# Don't do selective copy for multi-dimensional arrays (yet)
				elsif array.dimensions > 1
					minihash[:offset] = '0'
					minihash[:variable_dimensions] = array.size.join('*')
				end
				
				# Apply the mini-search-and-replace hash to create the memory copies from device to host
				processed[:mem_copy_D2H] += search_and_replace(minihash,skeletons[:mem_copy_D2H])
				if array.dimensions == 2 && @options[:target] == 'GPU-CUDA' && false
					processed[:mem_copy_D2H] += INDENT+'}'
				end
			end
			
			# Apply the search-and-replace hash to all timer skeletons and the host skeleton
			(['host']+TIMER_FILES).each do |skeleton|
				search_and_replace!(algorithm.hash,skeletons[skeleton.to_sym])
			end
			
			# Repair some invalid syntax that could have been introduced by performing the search-and-replace
			skeletons[:host].remove_extras
			
			# Run the prologue/epilogue code through the search-and-replace hash
			search_and_replace!(algorithm.hash,skeletons[:prologue])
			search_and_replace!(algorithm.hash,skeletons[:epilogue])
			
			# Construct the final host function, inluding the timers and memory copies
			if @scheduler
				host = skeletons[:prologue     ] + 
				       skeletons[:timer_2_start] + skeletons[:host         ] + skeletons[:timer_2_stop ] +
				       skeletons[:epilogue     ]
			else
				host = skeletons[:prologue     ] + 
				       skeletons[:timer_1_start] + processed[:mem_prologue ] + processed[:mem_copy_H2D ] +
				       skeletons[:timer_2_start] + skeletons[:host         ] + skeletons[:timer_2_stop ] +
				       processed[:mem_copy_D2H ] + processed[:mem_epilogue ] + skeletons[:timer_1_stop ] + 
				       skeletons[:epilogue     ]
			end
			
			# Generate code to replace the original code, including verification code if specified by the option flag
			verify_skeleton = File.read(File.join(@dir[:verify_library],'verify_results.c'))
			timer_start = (@options[:measurements]) ? File.read(File.join(@dir[:verify_library],'timer_start.c')) : ''
			timer_stop  = (@options[:measurements]) ? File.read(File.join(@dir[:verify_library],'timer_stop.c')) : ''
			replacement_code, original_definition, verify_definition = algorithm.generate_replacement_code(@options, verify_skeleton, @result[:verify_code], @prefix, timer_start, timer_stop)
			@result[:host_declarations].push(verify_definition)
			
			# Add a performance model to the original code
			#replacement_code.insert(0,algorithm.performance_model_code('model'))
			
			# Replace mallocs and frees in the original code with aligned memory allocations (only for CPU-OpenCL targets with zero-copy)
			if @options[:zero_copy] == 1 && @options[:target] == 'CPU-OPENCL-INTEL'
				@result[:original_code].search_and_replace_function_call(C::Variable.parse('malloc'),C::Variable.parse(VARIABLE_PREFIX+'malloc_128'))
				@result[:original_code].search_and_replace_function_call(C::Variable.parse('free'),C::Variable.parse(VARIABLE_PREFIX+'free_128'))
			end
			
			# Give the original main function a new name
			@result[:original_code].search_and_replace_function_definition('main',VARIABLE_PREFIX+'main')
			
			# Replace the original code with a function call to the newly generated code
			@result[:original_code].search_and_replace_node(algorithm.code,replacement_code)
			
			# The host code is generated, push the data to the output hashes
			accelerated_definition = 'void '+algorithm.name+'_accelerated('+algorithm.lists[:host_definition]+')'
			@result[:host_code_lists].push(@prefix+accelerated_definition+' {'+NL+host+NL+'}'+NL+NL)
			@result[:host_declarations].push(@prefix+accelerated_definition+';'+NL+@prefix+original_definition+';')
		end
		
		
		def generate_memory(type,copy,arrays,index)
			
			# Find the corresponding array
			arrays.each do |array|
				if array.name == copy.name && (array.direction == copy.direction || array.direction == INOUT)
					
					# Load the skeleton from the skeleton library
					type += copy.direction if type == 'copy'
					skeleton = File.read(File.join(@dir[:common_library],'mem_async_'+type+@extension))
					
					# Create the find-and-replace hash
					minihash = { :array               => copy.name,
					             :id                  => copy.id,
					             :index               => index.to_s,
					             :direction           => copy.direction,
					             :definition          => array.definition,
					             :type                => array.type_name,
					             :flatten             => array.flatten,
					             :offset              => '0',
					             :variable_dimensions => array.size.join('*'),
					             :state               => copy.deadline}
					
					# Instanstiate the skeleton and add it to the final result
					@result[:host_code_lists].push(search_and_replace(minihash,skeleton))
					
					# Add a forward declaration of this function
					@result[:host_declarations].push(copy.get_definition(array.definition,type))
					
					# Done
					return
				end
			end
		end
		
	end
	
end

