import pandas as pd

class ReadDbarPwf():
    def __init__(self, path):

        self.path  = path

        
        with open(path) as f:
            self.lines = f.readlines()

        self.l_DBAR = 24
        for idx, line in enumerate(self.lines):

            if 'DLIN' in line: self.l_DLIN  = idx - 2
        
        self._getdata()

    def _getdata (self):
            
        columns = ['BusID', 'BusName', 'Code']

        data = []
        for i in range(self.l_DBAR, self.l_DLIN + 1):
            bus = self.lines[i].strip().split()[0]
            nome = self.lines[i].strip().split()[2]
            code_ = self.lines[i].strip().split()[-2]

            if len(code_) < 8:
                code = 0
            else:
                code = code_[-1]

            data.append([bus,nome,code])

        try:
            self.data = pd.DataFrame(data, columns=columns)
            for col in self.data.columns:
                try:
                    self.data[col] = self.data[col].astype('float')
                except:
                    pass
        except:
            self.data = pd.DataFrame(data)
            print('Check the data or the columns')
            print(columns)


# path = 'D:/0 FERV/0 Dados PYTHON\V3A3F2_Rev2\DS20221001\Output/PTOPER_20221001_00h00_PatLeve_V3A3F2_E16_S00_rev2.pwf'
# cases = ReadDbarPwf(path)
# df = cases.data
# df.to_csv('DBAR.csv', sep=';', index=False)


