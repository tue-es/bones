
module Adarwin
	
	# This class represents a loop nest. The end goal is to annotate the loop nest
	# with the corresponding species information. If the loop nest cannot be
	# parallelised (if there are dependences), the species information is not
	# printed.
	#
	# This class contains methods to perform among others the following:
	# * Find all array references in the loop nest
	# * Merge found array references into another array reference
	# * Translate array references into species
	# * Perform dependence tests to check for parallelism
	# 
	class Nest
		attr_accessor :code, :species, :name, :verbose
		attr_accessor :fused, :removed
		attr_accessor :copyins, :copyouts
		attr_accessor :depth, :level, :id
		attr_accessor :reads, :writes
		attr_accessor :outer_loops
		
		# Method to initialise the loop nest. The loop nest is initialised with the
		# following variables:
		# * An identifier for the order/depth in which the nest appears (+level+)
		# * The loop nest body in AST form (+code+)
		# * A unique identifier for this loop nest (+id+)
		# * A human readable name for this loop nest (+name+)
		# * Whether or not verbose information should be printed (+verbose+)
		def initialize(level, code, id, name, verbose, fused=0)
			@depth = level.length
			@level = level
			@code = code
			@id = id
			@name = name+'_k'+(@id+1).to_s
			@verbose = verbose
			
			# Set the default values in case there are dependences
			@species = ''
			@fused = fused
			@removed = false
			@copyins = []
			@copyouts = []
			
			# Get all loops from the loop body and subtract the outer loops from all
			# loops to obtain the set of inner loops (loops in the body).
			@all_loops = @code.get_all_loops()
			@outer_loops = @code.get_direct_loops()
			@inner_loops = @all_loops - @outer_loops

			# Get all local variable declarations
			@var_declarations = @code.get_var_declarations()
			
			# Process the read/write nodes in the loop body to obtain the array
			# reference characterisations. The references also need to be aware of all
			# loop data and of any if-statements in the loop body.
			@references = @code.clone.get_accesses().map do |reference|
				Reference.new(reference,@id,@inner_loops,@outer_loops,@var_declarations,@verbose)
			end
		
			# Perform the dependence test. The result can be either true or false.
			# Proceed only if there are no dependences.
			# Don't perform the dependence test if this is a fused loopnest
			@has_dependences = (@fused > 0) ? false : has_dependences?
			if !@has_dependences && !@references.empty?
			
				# Merge array reference characterisations into other array references
				merge_references()
				
				# Translate array reference characterisations into species and ARC
				translate_into_species()
				translate_into_arc()
				
				# Set the copyin/copyout data from the array references
				@copyins = @references.select{ |r| r.tA == 'read' }
				@copyouts = @references.select{ |r| r.tA == 'write' }
			end
		end
		
		# Perform the algorithm to merge array reference characterisations into
		# merged array references. This method is a copy of the merging algorithm
		# as found in the scientific paper.
		# TODO: Complete this algorithm to match the scientific paper version.
		def merge_references
			@references.each do |ref1|
				@references.each do |ref2|
					if ref1 != ref2
					
						# Perform the checks to see if merging is valid
						if ref1.tN == ref2.tN && ref1.tA == ref2.tA && ref1.tS == ref2.tS
							
							# Merge the domain (ref2 into ref1)
							ref1.tD.each_with_index do |tD,i|
								tD.merge(ref2.tD[i])
							end
							
							# Merge the number of elements (ref2 into ref1)
							ref1.tE.each_with_index do |tE,i|
								tE.merge(ref2.tE[i])
							end
							
							# Delete ref2
							@references.delete(ref2)
							
							# Something has changed: re-run the whole algorithm again
							merge_references()
							return
						end
					end
				end
			end
		end
		
		# Method to translate the array reference characterisations into species.
		# The actual logic is performed within the Reference class. In this method,
		# only the combining of the separate parts is performed.
		def translate_into_species

			# Obtain the reads and writes
			@reads = @references.select{ |r| r.tA == 'read' }
			@writes = @references.select{ |r| r.tA == 'write' }
			
			# Create a 'void' access pattern in case there is no read or no write.
			# Else, set the species for the individual accesses.
			read_names = (@reads.empty?) ? ['0:0|void'] : @reads.map{ |r| r.to_species }
			write_names = (@writes.empty?) ? ['0:0|void'] : @writes.map{ |r| r.to_species }

			# Remove a 'full' access pattern in case there is a same 'shared' write pattern
			write_names.each do |write_name|
				write_parts = write_name.split(PIPE)
				if write_parts.last == 'shared'
					read_names.each do |read_name|
						read_parts = read_name.split(PIPE)
						if read_parts.last == 'full' && read_parts.first == write_parts.first
							read_names.delete(read_name)
						end
					end
				end
			end
			
			# Combine the descriptions (using Reference's +to_s+ method) into species
			species_in = read_names.uniq.join(' '+WEDGE+' ')
			species_out = write_names.uniq.join(' '+WEDGE+' ')
			@species = species_in+' '+ARROW+' '+species_out
		end
		
		# Method to translate the array reference characterisations into a string.
		def translate_into_arc
			@arc = @references.map{ |r| r.to_arc }.join(' , ')
		end
		
		# Perform the dependence test for the current loop nest. This method gathers
		# all pairs of array references to test and calls the actual dependence
		# tests. Currently, the dependence tests are a combination of the GCD test
		# and the Banerjee test.
		def has_dependences?
			
			# Gather all the read/write and write/write pairs to test
			to_test = []
			writes = @references.select{ |r| r.tA == 'write' }
			writes.each do |ref1|
				@references.each do |ref2|
					
					# Only if the array names are the same and they are not tested before
					if ref1.tN == ref2.tN && !to_test.include?([ref2,ref1])
						
						# Only if the array references are different (e.g. don't test
						# A[i][j+4] and A[i][j+4]).
						if (ref1.get_references != ref2.get_references)
							to_test << [ref1,ref2]
						end
					end
				end
			end
			
			# Test all pairs using the GCD and Banerjee tests
			#p to_test.map{ |t| t.map{ |r| r.to_arc }}
			to_test.uniq.each do |pair|
				dependence_test = Dependence.new(pair[0],pair[1],@verbose)
				if dependence_test.result
					return true
				end
			end
			return false
		end
		
		# Perform a check to see if the loop nest has species that are not just
		# formed from shared or full patterns. If so, there is no parallelism.
		def has_species?
			return false if @removed
			return false if @has_dependences
			return false if @species == ''
			return false if (@writes) && (@writes.select{ |a| a.pattern == 'shared' }.length > 3)
			only_full = (@reads) ? @reads.select{ |a| a.pattern != 'full' }.empty? : false
			only_shared = (@writes) ? @writes.select{ |a| a.pattern != 'shared' }.empty? : false
			return !(only_full && only_shared)
		end
		
		# Method to print the start pragma of a species.
		def print_species_start
			PRAGMA_DELIMITER_START+PRAGMA_SPECIES+' kernel '+@species+PRAGMA_DELIMITER_END
		end
		
		# Method to print the end pragma of a species.
		def print_species_end
			PRAGMA_DELIMITER_START+PRAGMA_SPECIES+' endkernel '+@name+PRAGMA_DELIMITER_END
		end
		
		# Method to print the start of an array reference characterisation (ARC).
		def print_arc_start
			PRAGMA_DELIMITER_START+PRAGMA_ARC+' kernel '+@arc+PRAGMA_DELIMITER_END
		end
		
		# Method to print the end of an array reference characterisation (ARC).
		def print_arc_end
			PRAGMA_DELIMITER_START+PRAGMA_ARC+' endkernel '+@name+PRAGMA_DELIMITER_END
		end
		
		# Method to print the copyin pragma.
		def print_copyins
			copys = @copyins.map{ |a| a.to_copy(2*a.id) }
			PRAGMA_DELIMITER_START+PRAGMA_SPECIES+' copyin '+copys.join(' '+WEDGE+' ')+PRAGMA_DELIMITER_END
		end
		
		# Method to print the copyout pragma.
		def print_copyouts
			copys = @copyouts.map{ |a| a.to_copy(2*a.id+1) }
			PRAGMA_DELIMITER_START+PRAGMA_SPECIES+' copyout '+copys.join(' '+WEDGE+' ')+PRAGMA_DELIMITER_END
		end
		
		# Method to check if the loop nest has copyins.
		def has_copyins?
			return !(copyins.empty?) && !(copyins.select{ |r| r.tD if !r.tD.empty? }.empty?)
		end
		
		# Method to check if the loop nest has copyouts.
		def has_copyouts?
			return !(copyouts.empty?) && !(copyouts.select{ |r| r.tD if !r.tD.empty? }.empty?)
		end
	end
	
end