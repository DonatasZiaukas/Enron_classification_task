

# building features for tfidf vecotizer
# loop in emails_by_address
import os 
import re
import pickle
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText  #SnowballStemmer 
from poi_email_addresses import poiEmails      # emails of all known poi's
from GroupedPoiEmails import GroupedPoiEmails

import email

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

path_to_emails = os.path.join(os.getcwd(),'emails_by_address')

list_of_emails =  os.listdir(path_to_emails)

word_data = {}



J = GroupedPoiEmails () # dictionary: key reffers to persons name and value is a list of emails
for name in data_dict:
    merge_text = []
    stp_words = []
    data_point = data_dict[name]
  
    
    if data_point['email_address'] == 'NaN':
        word_data.update({name: ' '})
        
        
    elif name in J:
        for ml in J[name]:
            
            from_email = '(?<=from_)' + ml
            for i in list_of_emails:
        
                # searching for a match of person's email address from project dataset 
                # in emails_by_address directory
                if re.search(from_email, i):
                    # if match, open the file for reading(file contains path's in maildir directory of specific emails)
                    with open(os.path.join(path_to_emails,i), "r") as email_records:
                
                        for line in email_records:
                        
                            #replacing path /enron_mail_20110402/ with a location in my pc 
                            line = line.replace('enron_mail_20110402', '/home/easypc/Documents/Udacity_nano_degree/EnronProject/ud120-projects/tools')
                            line = line.rstrip()
                    
                            with open(line, "r") as text:
                                #open email message                            
                            
                                msg = email.message_from_file(text)
                                match1 = re.findall('\w+',msg['X-From']) # considering names of email sender
                                match2 = re.findall('\w+',msg['X-To'])   # and recipient as stopwords
                                if len(match1) >= 2:
                                    #condition to ignore data that do not in include name in 'X-From' and 'X-To'
                                    stp_words.append(match1[0])
                                    stp_words.append(match1[1])
                                if len(match2) >=2:
                                    stp_words.append(match2[0])
                                    stp_words.append(match2[1])
                                   # condition for the email to be send to poi
                                            
                                parsed_text = parseOutText(text)
                                for item in stp_words:
                                    text = parsed_text.replace(item, '')
                                merge_text.append(text)
                                            
                                                
                       # poi_data.append(data_point['poi'])
        
        wrd = ' '.join([merge_text[i] for i in range(len(merge_text))])
        wrd = ''.join(i for i in wrd if not i.isdigit())
        word_data.update({name: wrd})
        
    else:
        #
        from_email = '(?<=from_)' + data_point['email_address']
        for i in list_of_emails:
        
            # searching for a match of person's email address from project dataset 
            # in emails_by_address directory
            if re.search(from_email, i):
                # if match, open the file for reading(file contains path's in maildir directory of specific emails)
                with open(os.path.join(path_to_emails,i), "r") as email_records:
                
                        for line in email_records:
                        
                            #replacing path /enron_mail_20110402/ with a location in my pc 
                            line = line.replace('enron_mail_20110402', '/home/easypc/Documents/Udacity_nano_degree/EnronProject/ud120-projects/tools')
                            line = line.rstrip()
                    
                            with open(line, "r") as text:
                                #open email message                            
                            
                                msg = email.message_from_file(text)
                                match1 = re.findall('\w+',msg['X-From']) # considering names of email sender
                                match2 = re.findall('\w+',msg['X-To'])   # and recipient as stopwords
                                if len(match1) >= 2:
                                    #condition to ignore data that do not in include name in 'X-From' and 'X-To'
                                    stp_words.append(match1[0])
                                    stp_words.append(match1[1])
                                if len(match2) >=2:
                                    stp_words.append(match2[0])
                                    stp_words.append(match2[1])
                                   # condition for the email to be send to poi
                                            
                                parsed_text = parseOutText(text)
                                for item in stp_words:
                                    text = parsed_text.replace(item, '')
                                merge_text.append(text)
                                            
                                               
                     
        
        wrd = ' '.join([merge_text[i] for i in range(len(merge_text))])
        wrd = ''.join(i for i in wrd if not i.isdigit())
        word_data.update({name: wrd})
        
        

                
                
pickle.dump( word_data, open("email_data.pkl", "w") )
             
            
            

            
            
            
            
