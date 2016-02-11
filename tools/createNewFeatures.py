# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:50:32 2015

@author: easypc
"""
import sys
sys.path.append("../tools/")



def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages == 'NaN' or all_messages == 'NaN' or all_messages == 0:
        fraction = 0
    else:
        fraction = float(poi_messages)/float(all_messages) 
        

    return fraction
MoneyMeans = {'salary': 208627.0, 'expenses': 39797.0, 'bonus': 675997.35,
              'other': 294745.53}
MoneyMedians = {'salary': 223199.0, 'expenses': 28128.0, 'bonus': 300000.0, 'other': 919.0}
def salaryVSotherIncome(salary, other):
   fraction = 0
   if salary == 'NaN' or other == 'NaN' or other == 0:
       fraction = 0
   else:
       if other < 1000:
           if other in MoneyMeans:
               fraction = MoneyMeans[other]
           else:
               #
               fraction = 0
       else:
           fraction = float(salary)/float(other)
       
   return fraction
    
    