Network benchmark dataset simulator with Django WebApp

Build instructions:
1.) Create a virtual environment (You can also use a newer Python version):
virtualenv -p python3.6 .

2.) Activate the virtual environment:
source bin/activate

3.) Install the requirements
pip install -r requirements.txt
sudo apt install seq-gen

4.) Change into the src folder
cd src

5.) Start the Django project
# this only needed to be done once in the beginning, no need to do that: django-admin.py startproject simulator .

6.) Apply the migrations
python manage.py migrate

7.) Create the superuser
# this step is optional: python manage.py createsuperuser

8.) Run the server locally
python manage.py runserver


Note: Whenever you are making changes to the code, you need to run this:
python manage.py makemigrations
python manage.py migrate

