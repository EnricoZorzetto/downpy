#!/usr/bin/python
import os
import time
import requests

start_time = time.time()

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)
    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (original_parsed.hostname != redirect_parsed.hostname) and \
                           redirect_parsed.hostname != self.AUTH_HOST and \
                           original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return


def download_my_urls(list_of_urls, out_dir, username, password):
    session = SessionWithHeaderRedirection(username, password)
    n_urls = len(list_of_urls)
    for iiu in range(n_urls):
        print(iiu)
        url = list_of_urls[iiu]
        path_and_name = os.path.join(out_dir, url[url.rfind('/') + 1:])
        try:
            # submit the request using the session
            response = session.get(url, stream=True)
            # print(response.status_code)
            # raise an exception in case of http errors
            response.raise_for_status()
            # save the file
            with open(path_and_name, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fd.write(chunk)
        except requests.exceptions.HTTPError as e:
            # handle any errors here
            print(e)

# data to download:
# years of observations I want to download:
# years = [2014, 2015, 2016, 2017]
# keep this file in the right folder, where you want to download the data

pass_file = 'user_password.txt'
usp = [line.rstrip('\n') for line in open(pass_file)
                                if line.strip(' \n') != '']
username = usp[0]
password = usp[1]

urls_folder = 'tmpa_urls'
url_files = os.listdir(urls_folder)
num_url_files = len(url_files)
out_dir = os.path.join('..', 'data','tmpa_raw_data')
if not os.path.exists(out_dir):
    print('download_tmpa_data WARNING: output folder not found-must create it!')
#if not os.path.exists(out_dir):
#    os.makedirs(out_dir)

print('number of url files is {}'.format(num_url_files))
for iiy in range(num_url_files):
    # out_dir = os.path.join('raw_data')
    file_urls = os.path.join(urls_folder, url_files[iiy])
    lines = [line.rstrip('\n') for line in open(file_urls)
                                    if line.strip(' \n') != '']
    print('number of urls in {} th url file is {}'.format(iiy, len(lines)))
    print(iiy)
    download_my_urls(lines, out_dir, username, password)


execution_time = time.time() - start_time
print("---execution time was %s minutes ---" % (execution_time/60))
