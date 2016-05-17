# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:11:13 2016

@author: Delirium
"""

import PyQt4
import json
import sqlite3 as lite
import urllib2
import Config
from datetime import time


#time.resolution(second=1)


#dbc = lite.connect("data.db")
#cur = dbc.cursor()
#cur.execute("create table my_ind (id varchar(15) primary key, desc varchar(50), value float, time integer)")
#dbc.commit()

resp = urllib2.urlopen("http://moex.com/iss/engines/stock/markets/index/securities.json?securities=%s" % ",".join(Config.indexes))
my_json = json.loads(resp.read())
for r in my_json["securities"]["data"]:
    print "%s \t%s\t\tvalue: %f" % (r[0], r[2], r[5])

dbc.close()