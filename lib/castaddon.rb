# Include the CAST gem, which provides a C99-parser. The gem
# is based on an intermediate representation in the form of
# an abstract syntax tree (AST). The gem provides C to AST
# and AST to C.
module C
end
require 'rubygems'
require 'cast'

# Include the extentions to the CAST gem provided. These
# extentions provide a significant amount of functionality
# for Bones and A-Darwin.
require 'castaddon/node_common.rb'
require 'castaddon/node_adarwin.rb'
require 'castaddon/node_bones.rb'
require 'castaddon/transformations.rb'
require 'castaddon/type.rb'
require 'castaddon/index.rb'

# Modify the NodeArray and NodeChain lists to output correct
# code when printed to a file.
class C::NodeList
	# Modify the 'to_s' method to output correct code when printed
	# to a file. Originally, it would separate instances of the list
	# with a ','. Instead, a newline command is added.
	def to_s
		self.join("\n")
	end
end

