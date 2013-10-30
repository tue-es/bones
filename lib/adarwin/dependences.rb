module Adarwin
	
	# This class represents the dependence tests. The dependence tests are not
	# objects as such, the use of a class might therefore be a bit out of place.
	# Instead, the class rather holds all methods related to dependence tests.
	#
	# For an M-dimensional access, the problem of dependence testing is reduced to
	# that of determining whether a system of M linear equations of the form
	# >>> a_1*I_1 + a_2*I_2 + ... + a_n*I_n = a_0
	# has a simultaneous integer solution satisfying the loop/if bounds given as
	# >>> min_k <= I_k <= max_k
	#
	# Currently, the following conservative tests are implemented:
	# * The GCD (greatest common divisor) test
	# * The Banerjee test
	#
	# In case the accesses are multi-dimensional, we perform a subscript-by-
	# subscript checking. In other words, we test each dimension separately
	# using the two tests. If we find a possible dependence in one dimension, we
	# conclude that there is a dependence.
	class Dependence
		attr_accessor :result
		
		# Method to initialise the dependence tests. This method actually already
		# computes all the dependence tests and stores the result in a class
		# variable. It takes as input the pair of accesses it needs to check for
		# dependences.
		def initialize(access1,access2,verbose)
			@verbose = verbose
			bounds = [access1.bounds,access2.bounds]
			
			# Iterate over the dimensions of the array reference
			results = []
			dimensions = [access1.indices.size,access2.indices.size].min
			for dim in 1..dimensions
				ref1 = access1.indices[dim-1]
				ref2 = access2.indices[dim-1]
				loop_vars = [access1.all_loops.map{ |l| l[:var] },access2.all_loops.map{ |l| l[:var] }]
				
				# Conclude directly that there is no dependence if the references are
				# exactly the same.
				if ref1 == ref2
					results << false
					next
				end
				
				# TODO: Include the step in the dependence tests
				#p access1.tS[dim-1]
				
				# Get all variables, a linear equation, and the corresponding conditions
				all_vars, equation, conditions = get_linear_equation(ref1,ref2,bounds,loop_vars)
				
				# Conclude directly that there is no dependence if the variables are not
				# dependent on the loops.
				if equation[:ak].empty?
					results << false
					next
				end
			
				# Perform the GCD test
				gcd_result = gcd_test(all_vars,equation)
				
				# End if the GCD test concludes that there are no dependences
				if gcd_result == false
					results << false
					
				# Continue with Banerjee if GCD concludes there might be dependences
				else
					ban_result = ban_test(all_vars,equation,conditions)
					results << ban_result
				end
			end
			
			# Combine the results for all dimensions
			if results.include?(true)
				@result = true
			else
				@result = false
			end
		end

		# This method implements the GCD test. The test is based on the computation
		# of the greatest common divisor, giving it its name. The GCD test is based
		# on the fact that a linear equation in the form of
		# >>> a_1*I_1 + a_2*I_2 + ... + a_n*I_n = a_0
		# has an integer solution if and only if the greatest common divisor of a_1,
		# a_2,...,a_n  is a divisor of a_0. The GCD test checks for this
		# divisability by performing the division and checking if the result is
		# integer.
		#
		# This method returns true if there is an integer solution, not necessarily
		# within the loop bounds. Thus, if the method returns true, there might be a
		# dependence. If the method returns false, there is definitely no dependence.
		#
		# TODO: If the result (+division+) is symbolic, can we conclude anything?
		def gcd_test(all_vars,equation)
			
			# Gather all the data to perform the test. Here, base represents a_0 and
			# data represents a_1,a_2,...,a_n.
			base = equation[:a0]
			data = equation[:ak]
			
			# Perform the greatest common divisor calculation and perform the division
			result = gcd(data)
			division = base/result.to_f
			
			# See if the division is integer under the condition that we can test that
			if result == 0
				gcd_result = false
			elsif division.class != Float
				gcd_result = true
			else
				gcd_result = (division.to_i.to_f == division)
			end
			
			# Display and return the result
			puts MESSAGE+"GCD-test '#{gcd_result}' ---> (#{base})/(#{result}) = #{division}, gcd(#{data})" if @verbose
			return gcd_result
		end

		# This method implements the Banerjee test. This test takes loop bounds into
		# consideration. The test is based on a linear equation in the form of
		# >>> a_1*I_1 + a_2*I_2 + ... + a_n*I_n = a_0
		# and loop bounds in the form of
		# >>> min_k <= I_k <= max_k
		#
		# The test proceeds as follows. First, the values a_k+ and a_k- are
		# computed. Also, the bounds min_k and max_k are calculated from the loop
		# conditions. Following, the test computes the extreme values 'low' and
		# 'high'. Finally, the test computes whether the following holds:
		# >>> low <= a_0 <= high
		# If this holds, there might be a dependence (method returns true). If this
		# does not hold, there is definitely no dependence (method returns false).
		def ban_test(all_vars,equation,conditions)
			
			# Pre-process the data to obtain the a_k+, a_k-, and lower-bounds and
			# upper-bounds for a_k (min_k and max_k).
			values = []
			equation[:ak].each_with_index do |a,index|
				values << {
					:ak_plus => (a >= 0) ? a : 0,
					:ak_min => (a <= 0) ? -a : 0,
					:min_k => conditions[index][:min],
					:max_k => conditions[index][:max]
				}
			end
			
			# Compute the extreme values 'low' and 'high'. This is done symbolically.
			low, high = "0", "0"
			values.each do |v|
				partial_low = simplify("
					(#{v[:ak_plus]}) * (#{v[:min_k]}) -
					(#{v[:ak_min]}) * (#{v[:max_k]})
				")
				low = simplify("(#{low}) + (#{partial_low})")
				partial_high = simplify("
					(#{v[:ak_plus]}) * (#{v[:max_k]}) -
					(#{v[:ak_min]}) * (#{v[:min_k]})
				")
				high = simplify("(#{high}) + (#{partial_high})")
			end
			
			# Perform the actual test: checking if low <= a_0 <= high holds. This is
			# implemented as two parts: check the lower-bound and check the upper-
			# bound.
			# FIXME: This method uses the +max+ which might make a guess.
			base = equation[:a0]
			test1 = (base.to_s == max(low,base.to_s))
			test2 = (high == max(base.to_s,high))
			ban_result = (test1 && test2)
			
			# Display and return the results
			puts MESSAGE+"Banerjee '#{ban_result}' ---> (#{test1},#{test2}), '(#{low} <= #{base} <= #{high})'" if @verbose
			return ban_result
		end

		# This method retrieves a linear equation from a pair of access. Accesses
		# are transformed into a linear equation of the form
		# >>> a_1*I_1 + a_2*I_2 + ... + a_n*I_n = a_0
		# Additionally, this method returns a list of all variables and a list of
		# loop bounds corresponding to the linear equation's variables.
		def get_linear_equation(access1,access2,bounds,all_loop_vars)
			equation = { :a0 => 0, :ak => [] }
			all_vars = []
			conditions = []
			hash = {}
			
			# Loop over the two accesses
			[access1,access2].each_with_index do |access,index|
				access = simplify(access.to_s)
				
				# Get the variables (I_1 ... I_n) and modify the access expression
				vars = get_vars(access).uniq
				loop_vars = get_loop_vars(vars,all_loop_vars[index])
				all_vars = (all_vars + vars).uniq
				vars.each do |var_name|
					access = access.gsub(/\b#{var_name}\b/,"hash[:#{var_name}]")
				end
				
				# Create a hash of all the variables. For now, this is just the name of
				# the variable. The values will be set later. This uses the 'symbolic'
				# library.
				vars.each do |var_name|
					if !hash[var_name.to_sym]
						hash[var_name.to_sym] = var :name => var_name
					end
					hash[var_name.to_sym].value = hash[var_name.to_sym]
				end
				
				# Find the constant term (a_0). This uses the +eval+ method together
				# with the 'symbolic' gem to compute the term.
				loop_vars.each do |var_name|
					hash[var_name.to_sym].value = 0
				end
				base = eval(access).value
				val = (index == 0) ? base : -base
				equation[:a0] = equation[:a0] + val
				
				# Find the other terms (a_1, a_2, ... a_n). This uses the +eval+ method
				# together with the 'symbolic' gem to compute the terms.
				loop_vars.each do |var_name|
					hash[var_name.to_sym].value = 1
					val = eval(access).value - base
					val = (index == 0) ? val : -val
					equation[:ak] << val
					hash[var_name.to_sym].value = 0
				end
				
				# Get the loop bound conditions corresponding to the linear equation's
				# variables.
				loop_vars.each do |var_name|
					conditions << bounds[index].select{ |c| c[:var] == var_name }.first
				end
			end
			return all_vars, equation, conditions
		end

		# Implementation of a GCD method with any number of arguments. Relies on
		# Ruby's default GCD method. In contrast to the normal gcd method, this
		# method does not act on a number, but instead takes an array of numbers as
		# an input.
		def gcd(args)
			val = args.first
			args.drop(1).each do |argument|
				val = val.gcd(argument)
			end
			return val
		end
		
		# Method to obtain all variables in an array reference that are also loop
		# variables.
		def get_loop_vars(vars,all_loop_vars)
			return vars & all_loop_vars
		end

		# Method to combine an array of integers in the form of a subtraction. For
		# example, given the input [a,b,c,d], the output will be (a-b-c-d).
		# TODO: Remove this method
		#def merge_subtract(args)
		#	val = args.first
		#	args.drop(1).each do |argument|
		#		val = val - argument
		#	end
		#	return val
		#end
		
	end
end