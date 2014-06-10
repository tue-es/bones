
module Bones
	# This class is based on the standard Array class. It is
	# meant to contain a list of elements of the Variable class.
	# In that sense, using the Array class will suffice. However,
	# this class extends the list with a small number of addi-
	# tional methods. These methods involve selecting a subset
	# of the list or sorting the list.
	class Variablelist < Array
		attr_accessor :representative
		
		# This method returns a subset of the list, based on the
		# argument +direction+ given. It either returns a list of
		# input variables or a list of output variables.
		def select(direction)
			array = Variablelist.new()
			self.each do |element|
				array.push(element) if ((direction == INPUT) && (element.input?)) || ((direction == OUTPUT) && (element.output?)) || (element.direction == INOUT)
			end
			return array
		end
		
		# Method to set a representative variable for this variable-
		# list. It is set based on the variable's species-name, e.g.
		# 'in0' or 'out2'.
		def set_representative(ids)
			@representative = select(ids.to_s.scan(/\D+/).join)[ids.to_s.scan(/\d+/).join.to_i]
		end
		
		# This method is a short-hand version to select a list of
		# input variables. It calls the +select+ method internally.
		def inputs
			select(INPUT)
		end
		
		# This method is a short-hand version to select a list of
		# output variables. It calls the +select+ method internally.
		def outputs
			select(OUTPUT)
		end
		
		# This method is a short-hand version to select a list of
		# input only variables. It calls the +select+ method
		# internally.
		def inputs_only
			self-select(OUTPUT)
		end
		
		# This method is a short-hand version to select a list of
		# input only variables. It calls the +select+ method
		# internally.
		def outputs_only
			self-select(INPUT)
		end
		
		# This method sorts the list of variables based on its
		# species' pattern (e.g. element or chunk). An alphabet
		# is based as an argument to this method to specify the
		# prefered order. This alphabet must be an array of strings.
		def sort_by(alphabet)
			clone = self.clone
			self.clear
			alphabet.each do |letter|
				clone.each do |array|
					self.push(array) if array.species.pattern == letter
				end
			end
		end
		
	end
	
end
