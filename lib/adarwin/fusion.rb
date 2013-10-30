
# Determine whether kernel fusion is legal (see algorithm in paper/thesis)
def fusion_is_legal?(a, b)
	(a.writes + a.reads).each do |x|
		(b.writes + b.reads).each do |y|
			if (x.tN == y.tN) && (x.tA == 'write' || y.tA == 'write')
				puts Adarwin::MESSAGE+"Evaluating #{x.to_arc} and #{y.to_arc} for fusion"
				if x.tD.to_s != y.tD.to_s || x.tE.to_s != y.tE.to_s || x.tS.to_s != y.tS.to_s
					puts Adarwin::MESSAGE+"Unable to fuse #{x.to_arc} and #{y.to_arc}"
					return false
				end
			end
		end
	end
	puts Adarwin::MESSAGE+"Applying fusion"
	return true
end


# Perform the kernel fusion transformations
def kernel_fusion(nests, settings)
	
	# Select
	candidates = nests.select{ |n| n.has_species? }
	
	# Iterate
	prev = nil
	candidates.each_with_index do |nest,nest_index|
		curr = nest
		if prev
			
			# Get the loop details
			loops_prev = prev.code.get_direct_loops
			loops_curr = curr.code.get_direct_loops
			if loops_prev.size != loops_curr.size
				puts Adarwin::MESSAGE+"Unable to apply fusion, loop count does not match"
				next
			end
			
			# Only proceed if fusion is legal for this combination
			if fusion_is_legal?(prev, curr)
				fused_code = []
			
				# Get the bodies 
				body_curr = get_body(loops_curr.size,curr.code.clone)
				body_prev = get_body(loops_prev.size,prev.code.clone)
				
				# Fuse everything together: include if-statements for non-matching loop bounds
				if settings == 1
					
					# Create new loops
					loops_target = []
					loops_prev.zip(loops_curr).each do |prevl,currl|
						raise_error("Unequal step count #{prevl[:step]} versus #{currl[:step]}") if prevl[:step] != currl[:step]
						minmin = exact_min(prevl[:min],currl[:min])
						maxmax = exact_max(prevl[:max],currl[:max])
						loop_datum = { :var => prevl[:var]+currl[:var], :min => minmin, :max => maxmax, :step => prevl[:step]}
						loops_target.push(loop_datum)
						
						# Replace all occurances of the fused loop variable in the current/previous codes
						body_prev = body_prev.replace_variable(prevl[:var],loop_datum[:var])
						body_curr = body_curr.replace_variable(currl[:var],loop_datum[:var])
						
						# Set minimum if-statement conditions
						body_prev = create_if(loop_datum[:var],minmin,prevl[:min],body_prev,'>=')
						body_curr = create_if(loop_datum[:var],minmin,currl[:min],body_curr,'>=')
						
						# Set maximum if-statement conditions
						body_prev = create_if(loop_datum[:var],maxmax,prevl[:max],body_prev,'<=')
						body_curr = create_if(loop_datum[:var],maxmax,currl[:max],body_curr,'<=')
					end
					
					# Generate the new code
					fused_code.push(code_from_loops(loops_target,[body_prev,body_curr]))
				
				# Create a prologue in case of mismatching loop bounds (experimental)
				elsif settings == 2
					
					# Generate the loop body
					loops_target = []
					loops_prev.zip(loops_curr).each do |prevl,currl|
						raise_error("Unequal step count #{prevl[:step]} versus #{currl[:step]}") if prevl[:step] != currl[:step]
						body_prev = body_prev.replace_variable(prevl[:var],prevl[:var]+currl[:var])
						body_curr = body_curr.replace_variable(currl[:var],prevl[:var]+currl[:var])
					end
					
					# Create the main loop nest
					loops_target = []
					loops_prev.zip(loops_curr).each do |prevl,currl|
						minmin = exact_min(prevl[:min],currl[:min])
						minmax = exact_min(prevl[:max],currl[:max])
						loop_datum = { :var => prevl[:var]+currl[:var], :min => minmin, :max => minmax, :step => prevl[:step]}
						loops_target.push(loop_datum)
					end
					fused_code.push(code_from_loops(loops_target,[body_prev,body_curr]))
					
					# Create the epilogue
					body = []
					loops_target = []
					loops_prev.zip(loops_curr).each do |prevl,currl|
						minmax = exact_min(prevl[:max],currl[:max])
						maxmax = exact_max(prevl[:max],currl[:max])
						loop_datum = { :var => prevl[:var]+currl[:var], :min => minmax, :max => maxmax, :step => prevl[:step]}
						loops_target.push(loop_datum)
						if prevl[:max] != currl[:max]
							body = (prevl[:max] == maxmax) ? [body_curr] : [body_prev]
						end
					end
					fused_code.push(code_from_loops(loops_target,body))
				end
				
				# Add the newly created code to the original code
				fused_code.each_with_index do |fused_codelet,nest_id|
					puts fused_codelet
					prev.code.insert_prev(fused_codelet)
				
					# Create a new nest
					nest = Adarwin::Nest.new(prev.level, fused_codelet, prev.id, prev.name.gsub(/_k(\d+)/,'_fused')+nest_id.to_s, prev.verbose, 1)
					nests.push(nest)
				end

				
				# Set the other nests as to-be-removed
				prev.removed = true
				curr.removed = true
			end
		end
	
		# Next nest
		prev = nest
	end
end

# Return the body of a loop nest
def get_body(num_loops,code)
	return code if num_loops == 0
	if code.first.for_statement? && code.first.stmt
		code = code.first
	end
	if code.for_statement? && code.stmt
		return get_body(num_loops-1,code.stmt.stmts)
	end
	raise_error("Not a perfect nested loop")
end

# Create an if-statement in front of a statement
def create_if(loop_var,reference_bound,loop_bound,code,condition)
	if reference_bound != loop_bound
		return C::Statement.parse("if(#{loop_var} #{condition} #{loop_bound}) { #{code.to_s} }")
	end
	return code
end

# Generate code from a combination of loops and statements (the body)
def code_from_loops(loops,statements)
	code = ""
	
	# Start of the loops
	definition = "int "
	loops.each do |loop_datum|
		increment = (loop_datum[:step] == '1') ? "#{loop_datum[:var]}++" : "#{loop_datum[:var]}=#{loop_datum[:var]}+#{loop_datum[:step]}"
		code += "for(#{definition}#{loop_datum[:var]}=#{loop_datum[:min]}; #{loop_datum[:var]}<=#{loop_datum[:max]}; #{increment}) {"
	end
	
	# Loop body
	statements.each do |statement|
		code += statement.to_s
	end
	
	# End of the loops
	loops.size.times{ |i| code += "}" }
	
	C::Statement.parse(code)
end