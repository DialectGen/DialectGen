import argparse
import logging
import boto3
import os

def main():
   """
   Code for creating/updating/deleting a qualification type at Amazon Mechanical Turk
   Important Note: Do not hard code the key and secret_key arguments
   """

   parser = argparse.ArgumentParser(description='Create qualification test for AAE-Dialect annotations')
   parser.add_argument('--aws_access_key_id', help='aws_access_key_id -- DO NOT HARDCODE IT')
   parser.add_argument('--aws_secret_access_key', help='aws_secret_access_key -- DO NOT HARDCODE IT')
   parser.add_argument('--questions', help='qualification questions (xml file)')
   parser.add_argument('--answers', help='answers to qualification questions (xml file)')
   parser.add_argument('--worker_id', help='worker id, if given we give worker access to the qualification type', \
               default=None)
   parser.add_argument('--Name',  help='name of qualification test', default='AAE Dialect Qualification Test')
   parser.add_argument('--Keywords', help='keywords that help worker find your test', \
               default='	dialect, african american, AAVE, african american english, AAE, african american vernacular english, dialectal bias')
   parser.add_argument('--Description', help='description of qualification test', \
                              default='This is a verification of whether you are qualified to annotate African American English dialect HITs.')
   parser.add_argument('--TestDurationInSeconds', help='time for workers to complete the test', default=30)
   parser.add_argument('--RetryDelayInSeconds',help='time workers should wait until they retake the test', default=10)
   parser.add_argument('--update', help='if true it updates an existing qualification type', action='store_true')
   parser.add_argument('--verbose', help='increase output verbosity', default=True)
   parser.add_argument('--delete', help='delete qualification type', action='store_true')
   parser.add_argument('--sandbox', help='create type in sandbox', action='store_true')
   args = parser.parse_args()

   if args.verbose:
      logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

   questions = open(args.questions, mode='r').read()
   answers = open(args.answers, mode='r').read()

   if args.sandbox:
      mturk = boto3.client('mturk',
         aws_access_key_id=args.aws_access_key_id,
         aws_secret_access_key=args.aws_secret_access_key,
         region_name='us-east-1',
         endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com')
   else:
      mturk = boto3.client('mturk',
         aws_access_key_id=args.aws_access_key_id,
         aws_secret_access_key=args.aws_secret_access_key,
         region_name='us-east-1',
         endpoint_url='https://mturk-requester.us-east-1.amazonaws.com')
      
   if not args.update:
      try:
         qual_response = mturk.create_qualification_type(
            Name=args.Name,
            Keywords=args.Keywords,
            Description=args.Description,
            QualificationTypeStatus='Active',
            Test=questions,
            AnswerKey=answers,
            RetryDelayInSeconds=args.RetryDelayInSeconds,
            TestDurationInSeconds=args.TestDurationInSeconds)
         qualification_type_id = qual_response['QualificationType']['QualificationTypeId']
         logging.info('Congrats! You have created a new qualification type')
         logging.info('You can refer to it using the following id: %s' % (qualification_type_id))
         logging.warning(' The qualification_type_id is saved under: qualification_type_id file.')
         logging.warning(' This is the id you will use to refer to your qualification test when creating your HIT!')
         
         q_id = open('qualification_type_id', 'w')
         q_id.write(qualification_type_id)
      except:
         logging.warning(' You have already created your qualification type. Read from qualification_type_id file...')
         try:
            q_id = open('qualification_type_id','r')
            qualification_type_id = q_id.readline()
         except:
            logging.error(' You have probably deleted the qualification type id file')
   else:
      logging.warning(' You have already created your qualification type. Read from qualification_type_id file...')
      try:
         q_id = open('qualification_type_id', 'r')
         qualification_type_id = q_id.readline()
         mturk.update_qualification_type(
            QualificationTypeId=qualification_type_id,
            Description=args.Description,
            Test=questions,
            AnswerKey=answers,
            RetryDelayInSeconds=args.RetryDelayInSeconds,
            TestDurationInSeconds=args.TestDurationInSeconds)
      except:
         logging.error(' You have probably deleted the qualification type id file')

   # If worker id is provided try to link to it
   if args.worker_id:
      mturk.associate_qualification_with_worker(
                     QualificationTypeId=qualification_type_id,
                     WorkerId=args.worker_id,
                     IntegerValue=0,
                     SendNotification=True)

      response = mturk.list_workers_with_qualification_type(
                     QualificationTypeId=qualification_type_id)

      logging.info(' You have associated your qualification type to the worker with id: %s ' % str(response))
   else:
      logging.info(' You may want to associate your qualification type to a worker or attach it to an HIT!')

        # Delete the qualification type
   if args.delete:
      try:
         q_id = open('qualification_type_id', 'r')
         qualification_type_id = q_id.readline()
         mturk.delete_qualification_type(QualificationTypeId=qualification_type_id)
         os.remove('qualification_type_id')
         logging.warning(' You have already created your qualification type. Read from qualification_type_id file...')
      except:
         logging.error(' You have probably deleted the qualification type id file')

if __name__ == "__main__":
   main()
