from loguru import logger 
from sys import stdout 

log_format = {
	'log_time':'<W><k>{time:HH-MM-DD hh:mm:ss}</k></W>',
	'log_file':'<C><r>{file:^13}</r></C>',
	'log_line':'<W><k>{line:03d}</k></W>',
	'log_level':'<E><r>{level:^10}</r></E>',
	'log_message':'<Y><k>{message:<50}</k></Y>'
}

logger.remove()
logger.add(
	sink=stdout, 
	level='TRACE',
	format='#'.join(list(log_format.values()))
)
