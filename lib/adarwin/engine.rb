
module Adarwin
	
	# This is the main 'engine' for the A-darwin algorithmic species extraction
	# tool. It contains methods to parse the command-line arguments, to run the
	# pre-processor, to insert the annotations, and to pretty print the final
	# output.
	# TODO: Add a syntax check by a normal compiler first (e.g. gcc)
	class Engine < Common
		
		# Initializes the engine and processes the command line arguments. This
		# method uses the 'trollop' gem to parse the arguments and to create a
		# nicely formatted help menu. This method additionally initializes a result-
		# hash and reads the contents of the source file from disk.
		#
		# ==== Command-line usage:
		#    adarwin --application <input> [OPTIONS]
		#
		# ==== Options:
		#        --application, -a <s>:   Input application file
		#  --no-memory-annotations, -m:   Disable the printing of memory annotations
		#    --mem-remove-spurious, -s:   Memcopy optimisation: remove spurious copies
		#    --mem-copyin-to-front, -f:   Memcopy optimisation: move copyins to front
		#    --mem-copyout-to-back, -b:   Memcopy optimisation: move copyouts to back
		#      --mem-to-outer-loop, -l:   Memcopy optimisation: move copies to outer loops
		#    --only-alg-number, -o <i>:   Only generate code for the x-th species (99 -> all) (default: 99)
		#                --version, -v:   Print version and exit
		#                   --help, -h:   Show this message
		#
		def initialize
			@result = {:original_code            => [],
			           :species_code             => []}
			
			# Parse the command line options using the 'trollop' gem.
			@options = Trollop::options do
				version 'A-darwin, part of Bones version '+File.read(ADARWIN_DIR+'/VERSION').strip+' (c) 2013 Cedric Nugteren, Eindhoven University of Technology'
				banner  NL+'A-darwin is an algorithmic species extraction tool. ' +
				        'For more information, see the README.rdoc file or visit the Bones/A-darwin website at http://parse.ele.tue.nl/bones/.' + NL + NL +
				        'Usage:' + NL +
				        '    adarwin --application <input> [OPTIONS]' + NL +
				        'using the following flags:'
				opt :application,           'Input application file',                                           :short => 'a', :type => String
				opt :no_memory_annotations, 'Disable the printing of memory annotations',                       :short => 'm', :default => false
				opt :mem_remove_spurious,   'Memcopy optimisation: remove spurious copies',                     :short => 'r', :default => false
				opt :mem_copyin_to_front,   'Memcopy optimisation: move copyins to front',                      :short => 'f', :default => false
				opt :mem_copyout_to_back,   'Memcopy optimisation: move copyouts to back',                      :short => 'b', :default => false
				opt :mem_to_outer_loop,     'Memcopy optimisation: move copies to outer loops',                 :short => 'l', :default => false
				opt :fusion,                'Type of kernel fusion to perform (0 -> disable)',                  :short => 'k', :type => Integer, :default => 0
				opt :print_arc,             'Print array reference characterisations (ARC) instead of species', :short => 'c', :default => false
				opt :silent,                'Become silent (no message printing)',                              :short => 's', :default => false
				opt :only_alg_number,       'Only generate code for the x-th species (99 -> all)',              :short => 'o', :type => Integer, :default => 99
			end
			Trollop::die 'no input file supplied (use: --application)'              if !@options[:application_given]
			Trollop::die 'input file "'+@options[:application]+'" does not exist'   if !File.exists?(@options[:application])
			@options[:name] = @options[:application].split('/').last.split('.').first
			@options[:no_memory_annotations] = true if @options[:print_arc]
			
			# Obtain the source code from file
			@source = File.open(@options[:application],'r'){|f| f.read}
			@basename = File.basename(@options[:application],'.c')
		end
		
		# Method to process a file and to output target code. This method calls all
		# the other methods, it is the main engine.
		#
		# ==== Tasks:
		# * Run the preprocessor to obtain algorithm information.
		# * Use the 'CAST' gem to parse the source into an AST.
		# * Call the code generator to perform the real work and produce output.
		def process
			
			# Run the preprocessor
			preprocessor = Adarwin::Preprocessor.new(@source)
			preprocessor.process
			@result[:header_code] = preprocessor.header_code
			
			# Set-up the CAST gem to include certain types
			# FIXME: What about other (user-defined?) types?
			parser = C::Parser.new
			parser.type_names << 'FILE'
			parser.type_names << 'size_t'
			
			# Parse the original source code into AST form (using CAST)
			original_ast = parser.parse(preprocessor.parsed_code)
			
			# Create an AST of the SCoP (using CAST) and save a backup
			scop_ast = C::Block.parse('{'+preprocessor.scop_code+'}')
			original_scop_ast = scop_ast.clone
			
			# Process the scop to identify the loop nests of interest and to find the
			# corresponding species. This is the method performing most of the work.
			@nests = []
			@id = 0
			populate_nests(scop_ast)
			
			# Remove inner-loop (nested) species. This removes all species that are
			# found within another species. For completeness, this might be desired in
			# some cases.
			# TODO: Make this an option
			@nests.each do |nest|
				if nest.has_species?
					remove_inner_species(get_children(nest))
				end
			end
			@nests.delete_if{ |n| n.removed }
			
			# Iterate over the nests/statements to optimize the copies. Currently, 
			# this will only look at loop nests with a depth of 1. Re-call the memory
			# copy optimisations method every time a change is made.
			# TODO: Investigate what the depth should be.
			basenests = @nests.select{ |n| n.depth == 1 }
			recursive_copy_optimisations(basenests,@options)
			
			# Kernel fusion is enabled (1,2,3,4) or disabled (0)
			if @options[:fusion] > 0
				# Test if fusion is legal and perform the actual transformation
				kernel_fusion(@nests, @options[:fusion])
			end
			
			# Delete the to-be-removed code (because of fusion)
			@nests.each do |nest|
				if nest.removed
					scop_ast.remove_once(nest.code)
				end
			end
			@nests.delete_if{ |n| n.removed }
			
			# Insert the species and memory copy annotations into the original code.
			# Don't do this if the user specified that he is not interested in the
			# memory copy annotations.
			insert_copies(scop_ast) unless @options[:no_memory_annotations]
			insert_species(scop_ast)
			
			# Create the modified SCoP and remove the quotes from the pragma's
			# FIXME: This is a hack for now, this has conflicts with strings in code
			modified_scop = INDENT+SCOP_START+NL+scop_ast.to_s+NL+INDENT+SCOP_END+NL
			modified_scop = modified_scop.gsub(PRAGMA_DELIMITER_START,'')
			modified_scop = modified_scop.gsub(PRAGMA_DELIMITER_END,'')
			
			# Print the result SCoP
			puts modified_scop if !@options[:silent]
			
			# Store the result
			@result[:species_code] = preprocessor.target_code.gsub(preprocessor.scop_code,modified_scop)
		end
		
		# This method writes the output code to a file.
		def write_output
			
			# Populate the species file
			# TODO: The filename is fixed, make this an optional argument
			File.open(File.join(@options[:application].rpartition('.').first+'_species'+'.c'),'w') do |target|
				target.puts @result[:species_code]
			end
		end
		
		# This method populates the Nest datastructure (recursively). It is the main
		# method to process the loop nests and fine the species information. It is
		# called recursively.
		def populate_nests(ast,level=[])
			
			# Only proceed if it is a loop
			if ast.block?
				
				# Create the new loop nests for the current depth level
				ast.stmts.each_with_index do |nest,index|
					new_level = level.clone.push(index)
					
					# Push the loop nest, but only if it is not disabled by options
					if @options[:only_alg_number].to_i == 99 || @options[:only_alg_number].to_i == (@id+1)
						
						# Only continue if the nest is an actual loop nest
						if nest.for_statement?
							@nests.push(Nest.new(new_level,nest,@id,@basename,!@options[:silent]))
						end
					end
					@id += 1
				end
				
				# Proceed to the next depth level.
				# TODO: Make it an option to only investigate the outer most level(s).
				ast.stmts.each_with_index do |nest,index|
					new_level = level.clone.push(index)
					if nest.stmt # && new_level == 0
						populate_nests(nest.stmt,new_level)
					end
				end
			end
		end
		
		# This method removes all species in the current loop nest (called
		# recursively). It assumes these species should be removed.
		def remove_inner_species(nests)
			nests.each do |nest|
				nest.copyins = []
				nest.copyouts = []
				nest.species = ''
				nest.removed = true
				children = get_children(nest)
				remove_inner_species(children) if children
			end
		end
		
		# Method to obtain the children of a nest
		def get_children(parent)
			children = []
			@nests.map do |nest|
				if parent.depth+1 == nest.depth
					if parent.level == nest.level.reverse.drop(1).reverse
						children << nest
					end
				end
			end
			return children
		end
		
		# This method iterates over the loop nests and inserts the species into the
		# original AST. It also inserts the synchronisation barries when needed, and
		# only if the user is interested in the memory copy annotations.
		def insert_species(scop_ast)
			
			# Find out where the synchronisation statements are needed
			sync_needed = []
			@nests.each do |nest|
				sync_needed << nest.copyins.map{ |c| c.get_sync_id }
				sync_needed << nest.copyouts.map{ |c| c.get_sync_id }
			end
			sync_needed = sync_needed.flatten.uniq
			
			# Insert the annotations into the code
			sync = 0
			@nests.each do |nest|
				sync = 2*nest.id
				
				# Insert the pre-kernel synchronisation barrier
				if sync_needed.include?(sync) && !@options[:no_memory_annotations]
					nest.code.insert_prev(C::StringLiteral.parse(PRAGMA_DELIMITER_START+PRAGMA_SPECIES+' sync '+(sync).to_s+PRAGMA_DELIMITER_END))
				end
				
				# Insert the pre-kernel species (start of species)
				if nest.has_species?
					to_print = (@options[:print_arc]) ? nest.print_arc_start : nest.print_species_start
					nest.code.insert_prev(C::StringLiteral.parse(to_print))
				end
				
				# Insert the post-kernel synchronisation barrier
				if sync_needed.include?(sync+1) && !@options[:no_memory_annotations]
					node = (nest.code.next && nest.code.next.string? && nest.code.next.val =~ /pragma species copyout/) ? nest.code.next : nest.code
					node.insert_next(C::StringLiteral.parse(PRAGMA_DELIMITER_START+PRAGMA_SPECIES+' sync '+(sync+1).to_s+PRAGMA_DELIMITER_END))
				end
				
				# Insert the post-kernel species (end of species)
				if nest.has_species?
					to_print = (@options[:print_arc]) ? nest.print_arc_end : nest.print_species_end
					location = nest.code
					location.insert_next(C::StringLiteral.parse(to_print))
				end
			end
		end
		
		# Iterate over the loop nests and insert the memory copy annotations into
		# the original AST.
		def insert_copies(scop_ast)
			@nests.each do |nest|
				if nest.has_copyins?
					nest.code.insert_prev(C::StringLiteral.parse(nest.print_copyins))
				end
				if nest.has_copyouts?
					nest.code.insert_next(C::StringLiteral.parse(nest.print_copyouts))
				end
			end
		end
		
	end
	
end

