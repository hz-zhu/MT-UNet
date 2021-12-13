import sys, copy, os, datetime, re, uuid, time, json, random, string

global global_random_seed
global global_ID
global_random_seed = int(time.time()*1e7)
global_ID = uuid.uuid4()
random.seed(global_random_seed)

def ls_file(path=os.getcwd()):
    assert isinstance(path, str), 'path type error @hzhu_gen::ls_file(path)'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return files

def ls_dir(path=os.getcwd()):
    assert isinstance(path, str), 'path type error @hzhu_gen::ls_dir(path)'
    files = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return files

def ls_all(path=os.getcwd()):
    assert isinstance(path, str), 'path type error @hzhu_gen::ls_all(path)'
    return os.listdir(path)

def str_criteria(word, criteria):
    assert isinstance(word, str), 'word type error @hzhu_gen::str_criteria(word, criteria)'
    if isinstance(criteria, str):
        if criteria in word: return True
        else: return False
    if isinstance(criteria, list):
        for item in criteria:
            J = str_criteria(word, item)
            if J==True: return True
        return False
    if isinstance(criteria, tuple):
        for item in criteria:
            J = str_criteria(word, item)
            if J==False: return False
        return True
    assert False, 'criteria type error @hzhu_gen::str_criteria(word, criteria)'

def ls_name(path=os.getcwd(), name=None):
    ls = ls_file(path)
    if name is None:
        return ls
    return [f for f in ls if str_criteria(f,name)]

def disp(data):
    print(to_str(data))
    
def to_str(data):
    try:
        return json.dumps(data, indent=4)
    except:
        return str(data)
    
def random_str(length=5):
    letters = string.ascii_letters+string.digits
    return ''.join(random.choice(letters) for i in range(length))

def extract_number(s):
    return [float(number) for number in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", s)]
    
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False
    
def read_file(name, path=os.getcwd()):
    file_name = path+'/'+name
    with open(file_name, 'r') as file:
        return file.read()

def read_file_by_line(name, path=os.getcwd()):
    file_name = path+'/'+name
    r = []
    with open(file_name, 'r') as file:
        for line in file:
            r.append(line)
    return r
    
class QuickTimer:
    
    def __init__(self):
        self.start_time = time.perf_counter()
    def __call__(self):
        return time.perf_counter()-self.start_time
    def start(self):
        self.__init__()
        
class QuickHelper:
    
    def __init__(self, path=None, name=None, ID_length=5):
        if path is None: self.path = os.getcwd()
        else: self.path = path
        if name is None: self.name = ''
        else: self.name = name
        
        self.ID_length = ID_length
        self.init()
        
    def init(self):
        counter = 0
        while True:
            self.ID = random_str(self.ID_length)
            J = create_folder(self.path+'/'+self.name+'_'+self.ID)
            if J:
                self.dir = self.path+'/'+self.name+'_'+self.ID
                break
            else:
                print('Folder already exists! Creating new ID.')
                counter += 1
            if counter>=10:
                self.ID_length += 1
                counter = 0
        self.timer = QuickTimer()
        
    def time_elapsed(self):
        return self.timer()
    
    def __call__(self):
        return self.dir
    
    def __str__(self):
        return '- QuickHelper:\n - ID = %s\n - path = %s\n - elapsed time = %f(sec)\n'%(self.ID, self.dir, self.time_elapsed())

    def summary(self):
        content = self.__str__()
        print(content)
        with open(self.dir+'/QuickHelper_summary.txt','w') as file:
            file.write(content)