
module Adarwin
	
	# This class represents an interval [a..b] including a and b. The class has
	# the following methods:
	# * Initialise the interval (+initialize+)
	# * Print the interval (+to_s+)
	# * Merge an interval with another interval (+merge+)
	# * Return the length of the interval (+length+)
	class Interval
		attr_accessor :a, :b
		
		# Initialise the interval. This method performs a comparison to see whether
		# a or b is the upper-bound. This comparison is based on guesses made by the
		# +compare+ method. This method uses loop information if needed.
		# FIXME: Uses the +compare+ method which might be based on a guess
		def initialize(a,b,loops)
			@loops = loops
			a = simplify(a.to_s)
			b = simplify(b.to_s)
			case compare(a,b,@loops)
				when 'lt' || 'eq' then @a = a; @b = b
				when 'gt' then @a = b; @b = a
				else @a = a; @b = b
			end
		end
		
		# Print the interval as a string (e.g. [4..9]).
		def to_s
			@a+RANGE_SEP+@b
		end
		
		# Merge this interval with another interval. This is based on a comparison
		# made by the +compare+ method, which is an approximation based on loop
		# information.
		# FIXME: Uses the +compare+ method which might be based on a guess
		def merge(other_interval)
			@a = case compare(@a,other_interval.a,@loops)
				when 'gt' || 'eq' then other_interval.a
				when 'lt' then @a
				else other_interval.a
			end
			@b = case compare(@b,other_interval.b,@loops)
				when 'gt' || 'eq' then @b
				when 'lt' then other_interval.b
				else @b
			end
		end
		
		# Method to compute the length of the interval. For example, the length of
		# [a..b] is equal to (b-a+1).
		def length
			simplify("(#{@b})-(#{a})+1")
		end
	end
	
end