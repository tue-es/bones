
# Bones/Aset require 'fileutils' from the Ruby standard library.
require 'fileutils'

# Bones/Aset use the 'trollop' gem to parse command line options.
require 'rubygems'
require 'trollop'
require 'symbolic'

# Extending the Ruby standard string class to support some
# additional methods. This includes a hack of the gsub! command.
class String #:nodoc:
	
	# Extend the Ruby string class to be able to chain 'gsub!'
	#-commands. This code is taken from the web.
	meth = 'gsub!'
	orig_meth = "orig_#{meth}"
	alias_method orig_meth, meth
	define_method(meth) do |*args|
		self.send(orig_meth, *args)
		self
	end
	
end

# Set the newline character
NL = "\n"
# Set the tab size (currently: 2 spaces)
INDENT = "\t"

# A string representing the combination character ('^') of a species.
WEDGE = '^'
# A string representing the production character ('->') of a species.
ARROW = '->'
# A string representing the pipe character ('|') of a species.
PIPE = '|'
# A string representing the colon character (':') to separate ranges in dimensions.
RANGE_SEP = ':'
# A string representing the comma character (',') to separate different ranges.
DIM_SEP = ','

# Value to assume a variable to be
ASSUME_VAL = '1000'


# Helper method to evaluate mathematical expressions, possibly containing
# symbols. This method is only used for readability, without it the code
# is functionally correct, but expressions might be larger than needed.
def simplify(expr)
	raise_error('Invalid expression to simplify') if !expr
	expr = expr.gsub(' ','')
	
	# Immediately return if there is an array index in the expression
	return expr if expr =~ /\[/
	
	# Handle min/max functions
	if expr =~ /max/ || expr =~ /min/
		return expr
	end
	
	# Get all the variables
	vars = get_vars(expr)
	
	# Set all the variables
	hash = {}
	vars.uniq.each do |var_name|
		hash[var_name.to_sym] = var :name => var_name
		expr = expr.gsub(/\b#{var_name}\b/,"hash[:#{var_name}]")
	end
	
	# Simplify the string using the 'symbolic' gem.
	symbolic_expr = eval(expr)
	
	# Return the result as a string
	return symbolic_expr.to_s
end

# Get the variables in an expression
def get_vars(expr)
	expr.split(/\W+/).reject{ |s| (s.to_i.to_s == s || s.to_f.to_s == s || s == "") }
end

# Solve a linear equality (work in progress)
def solve(equality,variable,forbidden_vars)
	return "" if equality == ""
	
	# Perform the subtitution of the current variable
	expr = '-('+equality.gsub('=','-(').gsub(/\b#{variable}\b/,"0")+'))'
	
	# Simplify the result
	result = simplify(expr)
	
	# Return the result or nothing (if it still contains forbidden variables)
	vars = get_vars(result)
	if vars & forbidden_vars == []
		return result
	else
		return ""
	end
end

# Find the maximum value of 2 expressions
def max(expr1,expr2,assumptions=[])
	return expr1 if expr2 == ""
	comparison = simplify("(#{expr1})-(#{expr2})")
	
	# Process the assumptions
	assumptions.each do |assumption|
		comparison = simplify(comparison.gsub(assumption[0],assumption[1]))
	end
	
	# Test to find the maximum
	if (comparison.to_i.to_s == comparison || comparison.to_f.to_s == comparison)
		return expr1 if (comparison.to_i == 0)
		return expr1 if (comparison.to_i > 0)
		return expr2 if (comparison.to_i < 0)
	else
	
		# Handle min/max functions
		if comparison =~ /max/ || comparison =~ /min/
			return "max(#{expr1},#{expr2})"
		end
		
		# Find the maximum based on a guess
		var = get_vars(comparison).first
		assumptions << [var,ASSUME_VAL]
		#puts "WARNING: Don't know how to find the max/min of '(#{expr1})' and '(#{expr2})', assuming: #{var}=#{ASSUME_VAL}"
		return max(expr1,expr2,assumptions)
	end
end

# Find the minimum value of 2 expressions (based on the max method)
def min(expr1,expr2)
	return expr1 if expr2 == ""
	s1 = simplify(expr1)
	s2 = simplify(expr2)
	comparison = simplify("(#{s1})-(#{s2})")
	
	# Handle min/max functions
	if comparison =~ /max/ || comparison =~ /min/
		return s1 if s2 =~ /^max\(#{s1},.*\)$/ || s2 =~ /^max\(.*,#{s1}\)$/
		return s2 if s1 =~ /^max\(#{s2},.*\)$/ || s1 =~ /^max\(.*,#{s2}\)$/
		return "min(#{expr1},#{expr2})"
	end
	
	# Run the 'max' method
	maximum = max(expr1,expr2)
	return (maximum == expr1) ? expr2 : ( (maximum == expr2) ? expr1 : maximum.gsub('max(','min(') )
end

# Find the exact maximum value of 2 expressions
def exact_max(expr1,expr2)
	return expr1 if expr1 == expr2
	comparison = simplify("(#{expr1})-(#{expr2})")
	if (comparison.to_i.to_s == comparison || comparison.to_f.to_s == comparison)
		return expr1 if (comparison.to_i == 0)
		return expr1 if (comparison.to_i > 0)
		return expr2 if (comparison.to_i < 0)
	else
		return "max(#{expr1},#{expr2})"
	end
end

# Find the exact minimum value of 2 expressions (based on the exact_max method)
def exact_min(expr1,expr2)
	return expr1 if expr1 == expr2
	maximum = exact_max(expr1,expr2)
	return (maximum == expr1) ? expr2 : ( (maximum == expr2) ? expr1 : maximum.gsub('max(','min(') )
end


# Return the absolute value (if possible)
def abs(expr)
	return expr.to_i.abs.to_s if expr.to_i.to_s == expr
	return expr
end

# Compare two expressions
def compare(expr1,expr2,loop_data,assumptions=[])
	comparison = simplify("(#{expr1})-(#{expr2})")
	
	# Handle min/max functions
	if comparison =~ /max/ || comparison =~ /min/
		return comparison
	end
	
	# Process the assumptions
	assumptions.each do |assumption|
		comparison = simplify(comparison.gsub(assumption[0],assumption[1]))
	end
	
	# Known comparison
	if (comparison.to_i.to_s == comparison || comparison.to_f.to_s == comparison)
		return 'eq' if (comparison.to_i == 0)
		return 'gt' if (comparison.to_i > 0)
		return 'lt' if (comparison.to_i < 0)
	else
		
		# Comparison based on loop data
		get_vars(comparison).each do |var|
			loop_data.each do |loop_datum|
				if loop_datum[:var] == var
					assumptions << [var,loop_datum[:min]]
					#puts "WARNING: Modifying expression '(#{expr1}) vs (#{expr2})', assuming: #{var}=#{loop_datum[:min]}"
					return compare(expr1,expr2,loop_data,assumptions)
				end
			end
		end
		
		# Comparison based on a guess
		var = get_vars(comparison).first
		assumptions << [var,ASSUME_VAL]
		#puts "WARNING: Don't know how to compare '(#{expr1})' and '(#{expr2})', assuming: #{var}=#{ASSUME_VAL}"
		return compare(expr1,expr2,loop_data,assumptions)
	end
end