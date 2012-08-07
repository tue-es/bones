# Include the test helper
require File.dirname(__FILE__) + '/../test_helper'

# Test class for the primitive class.
class TestCommon < Test::Unit::TestCase
	
	# Set the test up.
	def setup
		@common = Bones::Common.new
	end
	
	def test_brackets
		tests   = ['(4)','(var_16)','a+(5)','b1+(var*16)','a-(-4)']
		results = ['4'  ,'var_16'  ,'a+5'  ,'b1+(var*16)','a+4'   ]
		tests.each_with_index do |test,index|
			assert_equal(results[index], @common.simplify(test))
		end
	end
	
	def test_alu_constants
		tests   = ['4+1','4*(4+3)','a+5','b1+(3*11)','(6-12)-2','(12-6)*3','-2-2-2','a-a','a-b']
		results = ['5'  ,'28'     ,'a+5','b1+33'    ,'-8'      ,'18'      ,'-6'    ,'0'  ,'a-b']
		tests.each_with_index do |test,index|
			assert_equal(results[index], @common.simplify(test))
		end
	end
	
	def test_division_removal
		tests   = ['2/10','4*(2/1)','2/(1*4)']
		results = ['2/10','8'      ,'2/4'    ]
		tests.each_with_index do |test,index|
			assert_equal(results[index], @common.simplify(test))
		end
	end
	
	def test_division
		tests   = ['(2048/2)-1','4*(2/1)','2/2','2/(1*4)','var+(13/3)+(12/3)']
		results = ['1023'      ,'8'      ,'1'  ,'2/4'    ,'var+(13/3)+4'     ]
		tests.each_with_index do |test,index|
			assert_equal(results[index], @common.simplify(test))
		end
	end
	
	
	def test_general
		tests   = ['((3)-(2)+1)+0','((2+0)-(1)+1)','(((id/(1))%(2/1)))+2','(0+id/(2))+1']
		results = ['2'            ,'2'            ,'(id%2)+2'            ,'(id/2)+1']
		tests.each_with_index do |test,index|
			assert_equal(results[index], @common.simplify(test))
		end
	end
	
end

