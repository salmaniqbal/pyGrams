##
# pyGrams
#
# @file
# @version 0.1

.DEFAULT_GOAL := test

test:
	@cd tests
	@pytest tests/

# end
