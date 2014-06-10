
module Adarwin
	
	# This class represents an array reference characterisation. This reference is
	# constructed as a 5-tuple (tN,tA,tD,tE,tS) with the following information:
	# * tN: The name of the reference.
	# * tA: The access direction (read or write).
	# * tD: The full domain accessed.
	# * tE: The number of elements accessed each iteration (the size).
	# * tS: The step of a accesses among iterations.
	# To be able to compute the 5-tuple, the reference also stores information
	# about the loops and conditional statements to which the original array
	# reference is subjected.
	#
	# This class contains methods to perform among others the following:
	# * Initialise the class and sets the 5-tuple (N,A,D,E,S)
	# * Retrieve information on array indices
	# * Print in different forms (species, ARC, copy/sync pragma's)
	class Reference
		attr_accessor :tN, :tA, :tD, :tE, :tS
		attr_accessor :bounds, :indices, :pattern, :id
		attr_accessor :all_loops
		
		# This method initialises the array reference class. It takes details of the
		# reference itself and details of the loop nest it belongs to. The method
		# performs among others the following:
		# * It initialises the 5-tuple (N,A,D,E,S)
		# * It constructs the sets of loops (all,inner,outer) for this reference
		# * It computes the bounds based on loop data and on if-statements
		# * It computes the domain (D), number of elements (E), and step (S)
		def initialize(reference,id,inner_loops,outer_loops,var_declarations,verbose)
			@id = id
			
			# Initialise the 5-tuple (already fill in N and A)
			@tN = reference[:name]
			@tA = reference[:type]
			@tD = []
			@tE = []
			@tS = []
			
			# Set the inner loops as the loop nest's inner loop intersected with all
			# loops found for this statement. Beware of the difference between loops
			# of a loop nest and loops of a statement.
			@all_loops = reference[:loop_data]
			@inner_loops = inner_loops & @all_loops
			@outer_loops = outer_loops

			# Set the list of all local variables
			@var_declarations = var_declarations
			
			# Set the indices of the array reference (e.g. 2*i+4). The size of this
			# array is equal to the number of dimensions of the array.
			@indices = reference[:indices]
			
			# Set the if-statements for the reference. Process them together with the
			# loop start/end conditions to obtain a final set of conditions/bounds.
			@bounds = []
			loop_vars = @all_loops.map{ |l| l[:var]}
			@all_loops.each do |loop_data|
				conditions = [loop_data[:min],loop_data[:max]]
				reference[:if_statements].each do |if_statement|
					if !array_includes_local_vars(if_statement,loop_vars)
						condition_if = if_statement.map{ |c| solve(c,loop_data[:var],loop_vars) }
						conditions[0] = max(conditions[0],condition_if[0])
						conditions[1] = min(conditions[1],condition_if[1])
					end
				end
				@bounds << { :var => loop_data[:var], :min => conditions[0], :max => conditions[1] }
			end
			
			# Compute the domain (D) based on the bounds. The bounds are derived from
			# the if-statements and for-loops.
			@tD = @indices.map do |i|
				index_to_interval(i,@bounds)
			end
			
			# Compute the number of elements (E) accessed every iteration (the size).
			# TODO: Clean-up this method.
			@tE = @indices.map do |i|
				#if !dependent?(i,@all_loops)
				#	puts "independent"
				#	index_to_interval(i,@inner_loops)
				#else
					#puts "dependent"
					get_base_offset(i)
				#end
			end
			
			# Compute the step taken. There are 3 cases considered the index is: 1)
			# dependent on the outer loops, 2) dependent on the inner loops, or 3)
			# indepdent of any loops.
			@tS = @indices.map do |i|
				if dependent?(i,@inner_loops)
					index_to_interval(i,@inner_loops).length
				elsif dependent?(i,@outer_loops)
					get_step(i,@outer_loops)
				else
					'0'
				end
			end
			
			# If the step and the domain are equal in size, the step can also be set
			# to zero to reflect accessing the full array.
			@tS.each_with_index do |tS,index|
				if (tS == @tD[index].length) || (@tD[index].length == '1')
					@tS[index] = '0'
				end
			end

			# Check for local variables in the domain. If they exist ask the user to fill
			# in the bounds.
			# TODO: Make this a command-line question asked to the user. For now, several
			# known values are simply put here - for ease of automated testing.
			@tD.each do |bounds|

				# Bounds are equal (e.g. [t:t])
				if bounds.a == bounds.b && string_includes_local_vars(bounds.a)

					# Default (assume 'char')
					a = '0'
					b = '255'

					# Overrides (see notice above)
					b = 'NUM_CLUSTERS-1' if bounds.a == 'cluster_index'
					b = 'no_of_nodes-1' if bounds.b == 'id'

					# Output a warning
					puts WARNING+"Bounds of '#{bounds.a}' variable unknown, assuming #{a}:#{b}"
					bounds.a = a
					bounds.b = b

				# Not equal but both problematic
				elsif string_includes_local_vars(bounds.a) && string_includes_local_vars(bounds.b)

					# Default (assume 'char')
					a = '0'
					b = '255'

					# Overrides (see notice above)
					b = 'no_of_nodes-1' if bounds.a == 'val2'

					# Output a warning
					puts WARNING+"Bounds of '#{bounds.a}' and '#{bounds.b}' variables unknown, assuming #{a}:#{b}"
					bounds.a = a
					bounds.b = b

				end
			end

			
			# Print the result
			puts MESSAGE+"Found: #{to_arc}" if verbose
		end
		
		# This method replaces loop variables for a given set of loops with 0. This
		# basically gives us the offset of array references with respect to the loop
		# variable. For example, A[2*i+4] and A[i+j+3] will give us [4,j+3] with
		# repsect to an i-loop.
		def get_base_offset(index)
			index = index.clone
			@outer_loops.each do |for_loop|
				search = C::Variable.parse(for_loop[:var])
				replace = C::Expression.parse('0')
				index = index.search_and_replace_node(search,replace)
			end
			return index_to_interval(index,@inner_loops)
		end
		
		# Method to fill in the ranges for an array reference. This is based on
		# information of the loop nests. The output is an interval.
		def index_to_interval(index,loops)
			access_min = find_extreme(:min,index,loops)
			access_max = find_extreme(:max,index,loops)
			return Interval.new(access_min,access_max,@all_loops)
		end
		
		# Substitute loop data with the upper-bound or lower-bound of a loop to find
		# the minimum/maximum of an array reference. The body is executed twice,
		# because a loop bound can be based on another loop variable.
		def find_extreme(position,index,loops)
			index = index.clone
			2.times do
				loops.each do |for_loop|
					search = C::Variable.parse(for_loop[:var])
					replace = C::Expression.parse(for_loop[position])
					index = index.search_and_replace_node(search,replace)
				end
			end
			return simplify(index.to_s.gsub(';','').gsub(' ','').gsub("\t",''))
		end
		
		# Method to check whether the an index is dependent on a given set of loops.
		# For example, A[i+3] is independent of j, but dependent on i.
		def dependent?(index,loops)
			index.preorder do |node|
				if node.variable?
					loops.each do |for_loop|
						return true if (node.name == for_loop[:var])
					end
				end
			end
			return false
		end
		
		# Method to retrieve the step for a given array index and loops. The method
		# returns the difference between two subsequent iterations: one with the
		# loop variable at 0 and one after the first increment.
		def get_step(index,loops)
			
			# Replace the loop indices with 0
			index1 = index.clone
			loops.each do |for_loop|
				search = C::Variable.parse(for_loop[:var])
				replace = C::Expression.parse('0')
				index1 = index1.search_and_replace_node(search,replace)
			end
			
			# Replace the loop indices with the loop step
			index2 = index.clone
			loops.each do |for_loop|
				search = C::Variable.parse(for_loop[:var])
				replace = C::Expression.parse(for_loop[:step])
				index2 = index2.search_and_replace_node(search,replace)
			end
			
			# Return the difference
			return abs(simplify("(#{index2})-(#{index1})"))
		end
		
		# Method to output the result as algorithmic species. This reflects the
		# algorithm as presented in the scientific paper.
		def to_species
			if @tS.reject{ |s| s == "0"}.empty?
				if (@tA == 'read') # Full (steps length 0 and read)
					@pattern = 'full'
				else # Shared (steps length 0 and write)
					@pattern = 'shared'
				end
			elsif @tE.reject{ |s| s.length == "1"}.empty? # Element (sizes length 1)
				@pattern = 'element'
			elsif step_smaller_than_num_elements? # Neighbourhood (tS < tE)
				@pattern = 'neighbourhood('+@tE.join(DIM_SEP)+')'
			else # Chunk (tS >= tE)
				@pattern = 'chunk('+@tE.join(DIM_SEP)+')'
			end
			
			# Fill in the name and the domain and return the result
			return @tN+'['+@tD.join(DIM_SEP)+']'+PIPE+@pattern
		end
		
		# Method to output the result as an array reference characterisation (ARC).
		def to_arc
			return "(#{tN},#{tA},#{tD},#{tE},#{tS})".gsub('"','').gsub(' ','')
		end
		
		# Method to output a copyin/copyout statement. This indicates the name (N),
		# the domain (D), and a unique identifier.
		def to_copy(id)
			@tN+'['+@tD.join(DIM_SEP)+']'+'|'+id.to_s
		end
		
		# Method to print the unique identifier of the loop nest in terms of
		# synchronisation statements to be printed. This is a per-reference id
		# instead of a per-loop id, because it depends on the type of access (read
		# or write).
		def get_sync_id
			(@tA == 'write') ? 2*@id+1 : 2*@id
		end
		
		# Helper method for the +to_species+ method. This method compares the step
		# with the number of elements accessed to determine which one is smaller.
		# FIXME: This is based on the +compare+ method which might take a guess.
		def step_smaller_than_num_elements?
			@tS.each_with_index do |step,index|
				if step != '0'
					comparison = compare(step,@tE[index].length,@all_loops)
					if (comparison == 'lt')
						return true
					end
				end
			end
			return false
		end
		
		# Method to print out a human readable form of the array references (e.g.
		# [4*i+6][j]). This is basically what the +puts+ method also does.
		def get_references
			return @indices.to_ary.map{ |i| i.to_s }
		end
		
		# Method to find out if the reference is dependent on a variable. It is
		# used by the copy optimisations.
		def depends_on?(var)
			@indices.each do |index|
				index.preorder do |node|
					if node.variable?
						return true if (node.name == var)
					end
				end
			end
			return false
		end

		# Method to find if local variables are included
		def array_includes_local_vars(array, loop_vars)
			vars = @var_declarations - loop_vars
			array.each do |string|
				vars.each do |decl|
					if string =~ /\b#{decl}\b/
						return true
					end
				end
			end
			return false
		end
		def string_includes_local_vars(string)
			@var_declarations.each do |decl|
				if string =~ /\b#{decl}\b/
					return true
				end
			end
			return false
		end
		
	end
end