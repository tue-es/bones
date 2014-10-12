
module Bones
	# This class holds one algorithm, which includes a species,
	# a name, and the source C-code.
	#
	# The algorithm class holds all sorts of information on var-
	# iables. This information is only available after calling
	# the 'populate' method, which populates a lists of varia-
	# bles of all sorts: a regular list, a specialized hash,
	# and lists of input/output array variables.
	class Algorithm < Common
		attr_reader :name, :species, :code, :lists, :arrays, :id, :function_name
		attr_accessor :hash, :merge_factor, :register_caching_enabled
		
		# Constant to set the name of the algorithm's accelerated version
		ACCELERATED = '_accelerated'
		# Constant to set the name of the algorithm's original version
		ORIGINAL = '_original'
		
		# This method initializes the class. It gives the new
		# algorithm a name, species and source code. At initiali-
		# zation, this method checks if the name starts with a
		# digit. This is not allowed, so an underscore is added
		# prior to the digit.
		def initialize(name, filename, id, species, code)
			name = '_'+name if name =~ /^\d/
			@filename = filename
			@basename = name
			@name = (name+'_'+id).gsub(/\W/,'')
			@id = id
			@original_name = @name+ORIGINAL
			@accelerated_name = @name+ACCELERATED
			@species = species
			begin
				@code = C::Statement.parse(code).preprocess
			rescue
				@code = C::Statement.parse('{'+code+'}').preprocess
			end
			@hash = {}
			@lists = {:host_name => [],:host_definition => [], :argument_name => [], :argument_definition => [], :golden_name => []}
			@arrays = Variablelist.new()
			@constants = Variablelist.new()
			@merge_factor = 0
			@register_caching_enabled = 1
			@function_code = ''
			@function_name = ''
			
			# Set the initial hash
			@hash = {:algorithm_id        => @id,
			         :algorithm_name      => @name,
			         :algorithm_basename  => @basename,
			         :algorithm_filename  => @filename}
		end
		
		# This method sets the code and name for the function in
		# which the algorithm is found. This is done based on the
		# original code, which is given as input to this method.
		# The method does not return any value, instead, it sets
		# two class variables (@function_code and @function_name).
		def set_function(full_code)
			full_code.get_functions.each do |function|
				if function.node_exists?(@code)
					@function_code = function
					@function_name = function.name
				end
			end
			raise_error("Incorrect code found in body of #{@name}, something wrong with the classification?") if @function_code == ""
		end
		
		# This method performs the code transformations according
		# to the transformation settings as provided as an argument
		# to the function. It calls the various code transformation
		# functions as implemented for the CAST class. The resulting
		# modified code is finally stored in the search-and-replace
		# hash.
		# This method assumes that the populate method has already
		# been called, such that the hash contains the dimensions
		# needed to create the global ID definitions.
		def perform_transformations(transformation_settings)
			complexity = 0
			
			# Save the original code (with flattened arrays) in the hash as well
			new_code = @code.clone
			@arrays.each do |array|
				new_code.transform_flatten(array)
			end
			@hash[:algorithm_code0] = new_code.to_s
			
			# Loop over the number of transformation 'blocks'
			transformation_settings.split(' ').each_with_index do |transformation,num_transformation|
				new_code = @code.clone
				extra_indent = ''
				
				# Replace existing loops in the code (always do this)
				array = @arrays.representative
				array.species.dimensions.each_with_index do |dimension,num_dimension|
					index         =  (array.species.reverse?) ? num_dimension : array.species.dimensions.length-num_dimension-1
					index_reverse = !(array.species.reverse?) ? num_dimension : array.species.dimensions.length-num_dimension-1
					
					# Calculate the loop start and end conditions
					from = array.species.from_at(index)
					to = array.species.to_at(index)
					
					# Process the existing code and update the hash
					if from != to
						new_code, loop_variable_name = new_code.remove_loop(from,to)
						new_variable_name = GLOBAL_ID+'_'+index_reverse.to_s
						new_code.replace_variable(loop_variable_name,new_variable_name)
						update_hash(loop_variable_name)
					end
				end
				
				# Shuffle the indices of the first input(s) (conditionally do this)
				shuffle_arrays = []
				if transformation[0,1] == '2'
					shuffle_arrays.push(@arrays.select(INPUT)[0])
				elsif transformation[0,1] == '3'
					shuffle_arrays.push(@arrays.select(INPUT)[0])
					shuffle_arrays.push(@arrays.select(INPUT)[1])
				end
				new_code.transform_shuffle(shuffle_arrays)
				
				# Use the local on-chip memory (conditionally do this)
				if transformation[0,1] == '1'
					local_memory_arrays = [@arrays.select(INPUT)[0]]
					new_code.transform_use_local_memory(local_memory_arrays)
				end
				
				# Flatten the arrays to 1D (always do this)
				@arrays.each do |array|
					new_code.transform_flatten(array)
				end
				
				# Perform array substitution a.k.a. register caching (conditionally do this)
				if @register_caching_enabled == 1
					@arrays.outputs.each do |array|
						if array.species.element?
							if @arrays.inputs.include?(array)
								new_code.transform_substitution(array,true)
							else
								new_code.transform_substitution(array,false)
							end
							extra_indent = INDENT
						end
					end
				end
				
				# Perform transformations for reduction operations (conditionally do this)
				if transformation[1,1].to_i >= 1
					input = @arrays.select(INPUT)[0]
					@arrays.select(OUTPUT).each do |output|
						if output.species.shared?
							new_code = new_code.transform_reduction(input,output,transformation[1,1].to_i)
						end
					end
				end
				
				# Perform thread-merging (experimental)
				# TODO: Solve the problem related to constants (e.g chunk/example1.c)
				if @merge_factor == 0
					if transformation[0,1] == '4' && @hash[:parallelism].to_i >= 1024*1024
						@merge_factor = 4
					else
						@merge_factor = 1
					end
				end
				if @merge_factor > 1
					#puts @hash[:parallelism]
					if new_code.has_conditional_statements?
						puts MESSAGE+'Not coarsening ('+@merge_factor.to_s+'x) because of conditional statements in kernel body.'
					# TODO: Fix this temporary hack for multiple loops with mismatching bounds
					elsif ((@hash[:parallelism].to_i % @merge_factor) != 0) || (@hash[:parallelism].to_i == 4192256)
						puts MESSAGE+'Not coarsening ('+@merge_factor.to_s+'x) because of mismatching amount of parallelism ('+@hash[:parallelism]+').'
					else
						puts MESSAGE+'Coarsening threads by a factor '+@merge_factor.to_s+'.'
						
						# Update the hash
						@hash[:ids] = @hash[:ids].split(NL).map { |line|
							C::parse(line).transform_merge_threads(@merge_factor,[GLOBAL_ID]+@constants.map{ |c| c.name }).to_s.split(NL).each_with_index.map do |id,index|
								id.gsub(/\b#{GLOBAL_ID}\b/,"(#{GLOBAL_ID}+gridDim.x*blockDim.x*#{index})")
							end
						}.join(NL+INDENT*2)
						@hash[:parallelism] = (@hash[:parallelism].to_i / @merge_factor).to_s
						
						# Transform the code
						excludes = (@constants+@arrays).map { |c| c.name }
						new_code.transform_merge_threads(@merge_factor,excludes)
					end
				end
				
				# Obtain the complexity in terms of operations for the resulting code
				complexity += new_code.get_complexity
				
				# Store the resulting code in the hash
				resulting_code = new_code.strip_brackets.to_s
				@hash[('algorithm_code'+(num_transformation+1).to_s).to_sym] = (transformation[1,1].to_i >= 1) ? resulting_code : extra_indent+INDENT+resulting_code.gsub!(NL,NL+INDENT)
			end
			
			@hash[:complexity] = complexity.to_s
		end
		
		# This method creates the search-and-replace hash based on
		# information provided by the algorithm. It is called from
		# the 'populate' method of this class.
		#
		# == List of possible hash keys:
		#
		# algorithm_id
		#          _name
		#          _basename
		#          _filename
		#          _code*
		# (in*|out*)_type
		#           _name
		#           _devicename
		#           _devicepointer
		#           _dimensions
		#           _dimension*_to
		#                      _from
		#                      _sum
		#           _to
		#           _from
		#           _parameters
		#           _parameter*_to
		#                      _from
		#                      _sum
		#           _ids
		#           _localids
		#           _flatindex
		# (in|out)_names
		#         _devicenames
		#         _devicedefinitions
		#         _devicedefinitionsopencl
		# names
		# devicenames
		# devicedefinitions
		# devicedefinitionsopencl
		#
		# parallelism
		# factors
		# ids
		# verifyids
		#
		# argument_name
		# argument_definition
		# kernel_argument_list
		#
		def populate_hash
			@hash[:argument_name] = @lists[:argument_name]
			@hash[:argument_definition] = @lists[:argument_definition]
			
			# Obtain the necessary data for the hash per array
			parallelisms = []
			DIRECTIONS.each do |direction|
				arrays = @arrays.select(direction)
				arrays.each_with_index do |array,num_array|
					hashid = "#{direction}#{num_array}".to_sym
					
					# Gather the name and type data
					minihash = {:type          => array.type_name,
					            :name          => array.name,
					            :devicepointer => array.device_pointer,
					            :devicename    => array.device_name,
					            :flatindex     => array.flatindex}
					
					# Gather the dimensions data
					dimensions = array.species.dimensions
					dimensions.each_with_index do |dimension,num_dimension|
						minihash["dimension#{num_dimension}".to_sym] = {:sum  => simplify(sum(dimension)),
						                                                :from => simplify(from(dimension)),
						                                                :to   => simplify(to(dimension))}
					end
					minihash[:dimensions]  = simplify(dimensions.map { |d| sum(d) }.join('*'))
					minihash[:from] = dimensions.map { |d| from(d) }.zip(array.factors.drop(1).reverse).map { |e| simplify(e.join('')) }.join('+')
					minihash[:to  ] = dimensions.map { |d| to(d)   }.zip(array.factors.drop(1).reverse).map { |e| simplify(e.join('')) }.join('+')
					
					# Gather the parameter data
					if array.species.has_parameter?
						parameters = array.species.parameters
						parameters.each_with_index do |parameter,num_parameter|
							minihash["parameter#{num_parameter}".to_sym] = {:sum  => simplify(sum(parameter)),
							                                                :from => simplify(from(parameter)),
							                                                :to   => simplify(to(parameter))}
						end
						minihash[:parameters]  = simplify(parameters.map { |p| sum(p) }.join('*'))
					end
					
					# Store the data into the hash
					@hash[hashid] = minihash
					
					# Gather information regarding the parallelism
					if array.species.chunk?
						dim_div = simplify(minihash[:dimensions]+'/'+minihash[:parameters])
						parallelisms.push([dim_div,hashid,0])
					elsif array.species.element? || array.species.neighbourhood?
						parallelisms.push([minihash[:dimensions],hashid,1])
					end
					
					# Populate the global ID definitions hash, create the proper indices (and store as '{in/out}*_ids' in the hash)
					ids, localids, verifyids, factors = [], [], [], ['']
					dimensions = array.species.dimensions.clone
					dimensions.each_with_index do |dimension,num_dimension|
						index         =  (array.species.reverse?) ? num_dimension : array.species.dimensions.length-num_dimension-1
						index_reverse = !(array.species.reverse?) ? num_dimension : array.species.dimensions.length-num_dimension-1
						
						# Generate the index expressions
						divider = (array.species.chunk?) ? '/'+sum(array.species.parameters[index]) : ''
						dimensions_hash = (index == dimensions.length-1) ? '1' : dimensions.drop(index+1).map { |d| sum(d) }.join('*')
						dimensions_hash = simplify(dimensions_hash)
						dimensions_division = (dimensions_hash == '1') ? '' : '/('+dimensions_hash+')'
						minihash = {:dimensions1 => "#{GLOBAL_ID}#{dimensions_division}",
						            :dimensions2 => "#{LOCAL_ID }#{dimensions_division}",
						            :modulo      => (index_reverse != dimensions.length-1) ? '%('+simplify(sum(dimension)+divider)+')' : '',
						            :offset      => simplify(from(dimension))}
						expr_global = search_and_replace(minihash,"((<dimensions1>)<modulo>)+<offset>")
						expr_local  = search_and_replace(minihash,"((<dimensions2>)<modulo>)+<offset>")
						
						# Selectively push the ID definitions to the result array
						from = array.species.from_at(index)
						to = array.species.to_at(index)
						verifyids.push("const int #{GLOBAL_ID}_#{index_reverse} = "+expr_global+';')
						if from != to
							ids.push("const int #{GLOBAL_ID}_#{index_reverse} = "+expr_global+';')
							localids.push("const int #{LOCAL_ID }_#{index_reverse} = "+expr_local+';')
							factors.push(array.factors[index_reverse])
						end
					end
					
					# Store the results in the hash
					@hash[hashid][:ids] = ids.join(NL+INDENT*2)
					@hash[hashid][:localids] = localids.join(NL+INDENT*2)
					@hash[hashid][:verifyids] = verifyids.join(NL+INDENT*2)
					@hash[hashid][:factors] = factors.last
				end
				
				# Create lists of array names and definitions
				@hash["#{direction}_devicedefinitions".to_sym]       = arrays.map { |a| a.device_definition }.uniq.join(', ')
				@hash["#{direction}_devicedefinitionsopencl".to_sym] = arrays.map { |a| '__global '+a.device_definition }.uniq.join(', ')
				@hash["#{direction}_devicenames".to_sym]             = arrays.map { |a| a.device_name }.uniq.join(', ')
				@hash["#{direction}_names".to_sym]                   = arrays.map { |a| a.name }.uniq.join(', ')
			end
			@hash[:devicedefinitions]       = @arrays.map { |a| a.device_definition }.uniq.join(', ')
			@hash[:devicedefinitionsopencl] = @arrays.map { |a| '__global '+a.device_definition }.uniq.join(', ')
			@hash[:devicenames]             = @arrays.map { |a| a.device_name }.uniq.join(', ')
			@hash[:names]                   = @arrays.map { |a| a.name }.uniq.join(', ')
			
			# Set the parallelism for the complete species, first sort them according to priorities and then find the maximum
			# TODO: Remove the 'reverse' statement and get the 'ids' part working correctly for chunks
			# TODO: How to find the maximum of symbolic expressions?
			parallelisms = parallelisms.reverse.sort_by { |p| p[2] }
			parallelism = parallelisms.reverse.max_by { |p| p[0].to_i }
			@hash[:parallelism] = parallelism[0]
			@hash[:ids]         = @hash[parallelism[1]][:ids]
			@hash[:factors]     = @hash[parallelism[1]][:factors]
			@arrays.set_representative(parallelism[1])
		end
		
		# Helper function to create a the special code which is required
		# for OpenCL function calls to be able to use kernel arguments.
		def opencl_arguments(list,kernel_id)
			return '' if list == ''
			argument_string = ''
			list.split(', ').each_with_index do |variable,id|
				argument_string += 'clSetKernelArg(bones_kernel_'+@name+'_'+kernel_id.to_s+',bones_num_args+'+id.to_s+',sizeof('+variable.strip+'),(void*)&'+variable.strip+');'+NL+INDENT
			end
			return argument_string
		end
		
		# This method updates the hash after loops are removed from
		# the code. It takes as an argument a loop variable, which
		# it removes from both the ':argument_name' and ':argument_
		# definition' hash entries.
		def update_hash(loop_variable)
			names = @hash[:argument_name].split(', ')
			definitions = @hash[:argument_definition].split(', ')
			# TODO: The following two lines give problems with correlation-k4
			names.delete(loop_variable.to_s)
			definitions.each { |definition| definitions.delete(definition) if definition =~ /\b#{loop_variable}\b/ }
			@hash[:argument_name] = names.join(', ')
			@hash[:argument_definition] = definitions.join(', ')
			
			# Now, generate the special code which is required for OpenCL function calls to be able to use kernel arguments.
			@hash[:kernel_argument_list] = opencl_arguments([@hash[:devicenames],@hash[:argument_name]].join(', ').remove_extras,0)
			@hash[:kernel_argument_list_in] = opencl_arguments(@hash[:in_devicenames],0)
			@hash[:kernel_argument_list_out] = opencl_arguments(@hash[:out_devicenames],0)
			@hash[:kernel_argument_list_constants] = opencl_arguments(@hash[:argument_name],0)
			
			# Add declarations for the loop variables for the original code in the hash
			@hash[:algorithm_code0] = INDENT+"int #{loop_variable};"+NL+@hash[:algorithm_code0]
		end
		
		# Method to create a list of variables for the current
		# algorithm. These variables should hold two conditions:
		# 1) they are not local to the algorithm's code, and 2),
		# they are used in the algorithm's code.
		#
		# The method gets a lists of undefined variables in the
		# algorithm's code and subsequently searches the original
		# code for the definition of this variable.
		def populate_variables(original_code,defines)
			@code.undefined_variables.each do |name|
				type = @function_code.variable_type(name)
				raise_error('Variable '+name+' not declared in original code') if !type
				size = original_code.size(name)
				direction = @code.direction(name)
				size.map! { |s| simplify(replace_defines(s,defines)) }
				variable = Variable.new(name,type,size,direction,@id,@species.shared?)
				(variable.dimensions > 0) ? @arrays.push(variable) : @constants.push(variable)
			end
			raise_error('No input nor output arrays detected, make sure they are properly defined') if arrays.empty?
			
			DIRECTIONS.each do |direction|
				species = @species.structures(direction)
				if direction == INPUT && @species.shared?
					arrays = @arrays.inputs_only
				else
					arrays = @arrays.select(direction)
				end
				if !arrays.empty?
					
					# Check if the amount of input/ouput arrays is equal to the amount of input/output species
					if species.length < arrays.length
						array_names = arrays.map { |a| a.name }.join('","')
						raise_error(direction.capitalize+'put array count mismatch (expected '+species.length.to_s+', found '+arrays.length.to_s+' ["'+array_names+'"])')
					end

					# Set the species for the arrays (distinguish between arrays with and without a name)
					species.each do |structure|
						
						# Loop over all found arrays and match it with a species
						array = nil
						arrays.each do |free_array|
							if !free_array.species
								if structure.has_arrayname?
									if structure.name == free_array.name
										array = free_array
										break
									end
								else
									array = free_array
									break
								end
							end
						end
						
						# Still haven't found anything, assign the species to an array of equal name
						if !array
							arrays.each do |free_array|
								array = free_array if structure.name == free_array.name
							end
						end

						# Still haven't found anything, raise an error
						if !array
							raise_error("Could not find a matching array in C-code for a species with name '#{species.first.name}'")
						end
						
						# Process the assignment
						array.species = structure
						raise_error("Species of '#{array.species.name}' is mismatched with array '#{array.name}'") if array.species.name != array.name
						
						# Check if the array size was set, if not, it will be set to the species' size
						if array.size.empty?
							array.size = array.species.dimensions.map { |d| sum(d) }
							array.guess = true
							puts WARNING+'Could not determine size for array "'+array.name+'" automatically, assuming: '+array.size.inspect+'.'
						end
						
						# Set the multiplication factors (for later)
						array.set_factors
					end
				end
			end
			
			# Sort the arrays according to the alphabet
			if @arrays.length > 1
				@arrays.sort_by(['chunk','neighbourhood','element','shared','full'])
			end
		end
		
		# Method to populate 5 lists with variable information.
		# Below are listed the names of the four lists with an
		# example value:
		# 
		# host_name::           Example: 'array'
		# host_definition::     Example: 'int array[10][10]'
		# argument_name::       Example: 'threshold'
		# argument_definition:: Example: 'float threshold'
		# golden_name::         Example: 'golden_array'
		def populate_lists
			@constants.each do |variable|
				@lists[:host_name]          .push(variable.name)
				@lists[:host_definition]    .push(variable.definition)
				@lists[:argument_name]      .push(variable.name)
				@lists[:argument_definition].push(variable.definition)
				@lists[:golden_name]        .push(variable.name)
			end
			@arrays.each do |variable|
				@lists[:host_name]          .push(variable.name)
				@lists[:host_definition]    .push(variable.definition)
				@lists[:golden_name]        .push(variable.golden_name)
			end
			@lists.each { |name,list| @lists[name] = list.join(', ') }
		end
		
		# This method is used to generate verification code. This
		# verification code contains a copy of the original code.
		# It also provides a verification which compares the output
		# of the original code with the output of the generated
		# code. The verification code prints warnings if the outputs
		# are not equal, else it prints a success message.
		def generate_replacement_code(options, skeleton, verify_code, prefix, timer_start, timer_stop)
			replacement = C::NodeArray.new
			replacement.push(C::ExpressionStatement.parse(@accelerated_name+'('+@lists[:host_name]+');'))
			original_definition = ''
			verify_definitions = []
			if options[:verify]
				guesses = @arrays.map { |array| array.guess }
				if guesses.include?(true)
					puts WARNING+'Verification not supported for this class'
				else
					
					# Generate the replacement code and the original function
					@arrays.each do |array|
						replacement.insert(0,C::ExpressionStatement.parse("memcpy(#{array.golden_name},#{array.name},#{array.size.join('*')}*sizeof(#{array.type_name}));"))
						replacement.insert(0,C::Declaration.parse(array.definition.gsub!(/\b#{array.name}\b/,array.golden_name)+array.initialization))
					end
					replacement.push(C::ExpressionStatement.parse(@original_name+'('+@lists[:golden_name]+');'))
					original_definition = "void #{@original_name}(#{@lists[:host_definition]})"
					body = "#{timer_start}#{NL}  // Original code#{NL}#{@code}#{NL}#{timer_stop}"
					verify_code.push(prefix+original_definition+' {'+NL+body+'}'+NL+NL)
					@arrays.select(OUTPUT).each do |array|
						replacement.push(C::ExpressionStatement.parse(("bones_verify_results_#{array.name}_#{@id}(#{array.name}#{array.flatten},#{array.golden_name}#{array.flatten},#{@hash[:argument_name]});").remove_extras))
					end
					@arrays.each do |array|
						replacement.push(C::ExpressionStatement.parse("free(#{array.golden_name});")) if array.dynamic?
					end
					
					# Generate the verification function itself
					@arrays.select(OUTPUT).each_with_index do |array,num_array|
						minihash = @hash["out#{num_array}".to_sym]
						minihash[:name] = minihash[:name]+'_'+@id
						minihash[:argument_definition] = @hash[:argument_definition]
						instantiated_skeleton = search_and_replace(minihash,skeleton)
						verify_definitions.push(instantiated_skeleton.scan(/#{START_DEFINITION}(.+)#{END_DEFINITION}/m).join.strip.remove_extras)
						verify_code.push(instantiated_skeleton.remove_extras.gsub!(/#{START_DEFINITION}(.+)#{END_DEFINITION}/m,''))
					end
				end
			end
			return replacement, original_definition, verify_definitions.join(NL)
		end
		
		# Method to generate performance modeling code.
		# This method is still under construction and will not be called yet.
		# TODO: Complete this method
		def performance_model_code(model_dir)
			
			# Load the profile database
			profiles = Array.new
			File.read(File.join(model_dir,'profile.txt')).each do |line|
				profiles.push(line.split(','))
			end
			
			# Iterate over all the profiles
			result = C::NodeArray.new
			profiles.each do |profile|
				
				# Fill the hash with profile information and species information
				mini_hash = {
					:name => profile[0].strip,
					:comp => profile[1].strip,
					:coal => profile[2].strip,
					:unco => profile[3].strip,
					:copy => profile[4].strip,
					:f    => @hash[:complexity],
					:w    => @hash[:parallelism],
					:c    => @species.all_structures.map { |s| simplify('4*('+s.dimensions.map { |d| sum(d) }.join('*')+')') }.join(' + '),
					:m    => '1',
					:u    => '0',
					:o    => '8'
				}
				
				# Load the skeleton for the performance model and set the values according to the hash
				model_skeleton = File.read(File.join(model_dir,'model.c'))
				search_and_replace!(mini_hash,model_skeleton)
				result.push(C::Block.parse(model_skeleton))
			end
			return result
		end
	end
	
end

