

# Recursive copy optimisations
def recursive_copy_optimisations(nests,options)
	2.times do
		perform_copy_optimisations1(nests,options)
		perform_copy_optimisations2(nests,options)
		nests.each do |nest|
			children = get_children(nest)
			recursive_copy_optimisations(children,options) if !children.empty?
		end
		perform_copy_optimisations3(nests,options)
		perform_copy_optimisations3(nests,options)
	end
end

# First set of copyin/copyout optimisations (recursive)
def perform_copy_optimisations1(nests,options)
	previous = nil
	nests.each_with_index do |nest,nest_index|
		current = nest
		if previous
	
			# Remove spurious copies (out/in)
			if options[:mem_remove_spurious]
				previous.copyouts.each do |copyout|
					current.copyins.each do |copyin|
						if copyout.tN.to_s == copyin.tN.to_s && copyout.tD.to_s == copyin.tD.to_s
							current.copyins.delete(copyin)
							return perform_copy_optimisations1(nests,options)
						end
					end
				end
			end
	
			# Remove spurious copies (out/out)
			if options[:mem_remove_spurious]
				previous.copyouts.each do |copyout|
					current.copyouts.each do |other_copyout|
						if copyout.tN.to_s == other_copyout.tN.to_s && copyout.tD.to_s == other_copyout.tD.to_s
							previous.copyouts.delete(copyout)
							return perform_copy_optimisations1(nests,options)
						end
					end
				end
			end
		
			# Move copyins to the front
			if options[:mem_copyin_to_front]
				current.copyins.each do |copyin|
					if previous.writes && !previous.writes.map{ |w| w.tN }.include?(copyin.tN)
						previous.copyins.push(copyin)
						current.copyins.delete(copyin)
						return perform_copy_optimisations1(nests,options)
					end
				end
			end
		
		end
		
		# Next nest
		previous = nest
	end
end

# Second set of copyin/copyout optimisations (non-recursive)
def perform_copy_optimisations2(nests,options)
	nests.each_with_index do |nest,nest_index|
		current = nest
			
		# Move copyouts to the back
		if options[:mem_copyout_to_back]
			current.copyouts.each do |copyout|
				nests.each_with_index do |other_nest,other_nest_index|
					if other_nest.id > nest.id && other_nest.depth == nest.depth
						if other_nest.writes && !other_nest.writes.map{ |w| w.tN }.include?(copyout.tN)
							copyout.id = copyout.id+1
						else
							break
						end
					end
				end
			end
		end
		
		# Remove spurious copies (double in)
		if options[:mem_remove_spurious]
			current.copyins.each_with_index do |copyin,index|
				current.copyins.each_with_index do |other_copyin,other_index|
					if index != other_index
						if copyin.tN.to_s == other_copyin.tN.to_s && copyin.tD.to_s == other_copyin.tD.to_s
							if copyin.id > other_copyin.id
								current.copyins.delete(copyin)
							else
								current.copyins.delete(other_copyin)
							end
						end
					end
				end
			end
		end
		
	end
end

# Third set of copyin/copyout optimisations (inter-level)
def perform_copy_optimisations3(nests,options)
	nests.each do |nest|
		current = nest
		children = get_children(nest)
		if !children.empty?
			
			# Inter-level loop optimisations (move to outer loop)
			if options[:mem_to_outer_loop]
				
				# Move copyouts to outer loops
				max_id = children.map{ |c| 2*c.id+1 }.max
				children.each do |child|
					child.copyouts.each do |copyout|
						to_outer_loop = true
						nest.outer_loops.map{ |l| l[:var] }.each do |var|
							to_outer_loop = false if copyout.depends_on?(var)
						end
						children.each do |other_child|
							to_outer_loop = false if other_child.copyins.map{ |c| c.tN }.include?(copyout.tN)
						end
						to_outer_loop = false if copyout.get_sync_id < max_id
						if to_outer_loop
							copyout.id = nest.id
							nest.copyouts.push(copyout)
							child.copyouts.delete(copyout)
						end
					end
				end
			
				# Move copyins to outer loops
				children.first.copyins.each do |copyin|
					to_outer_loop = true
					nest.outer_loops.map{ |l| l[:var] }.each_with_index do |var,lindex|
						if copyin.depends_on?(var)
							to_outer_loop = false
							if copyin.tD[0].a == var && copyin.tD[0].b == var
								loopinfo = nest.outer_loops[lindex]
								if loopinfo[:step] == "1"
									copyin.tD[0].a = loopinfo[:min]
									copyin.tD[0].b = loopinfo[:max]
									to_outer_loop = true
								end
							end
						end
					end
					children.drop(1).each do |child|
						to_outer_loop = false if child.copyins.map{ |c| c.tN }.include?(copyin.tN)
						to_outer_loop = false if child.copyouts.map{ |c| c.tN }.include?(copyin.tN) && child != children.last
					end
					if to_outer_loop
						nest.copyins.push(copyin)
						children.first.copyins.delete(copyin)
					end
				end
				
			end
		end
	end
end