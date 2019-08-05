import csv,sys
from collections import defaultdict
#from yaml import CLoader as Loader

#remove html tags in strings
def unescape(s):
    if sys.version_info >= (3, 0):
        import html
        output = html.unescape(str(s))
    else:
        import htmllib

        p = htmllib.HTMLParser(None)
        p.save_bgn()
        try:
            p.feed(s)
        except:
            return s
        output=p.save_end()
    return output

#add tags to dict depending on the position of the brand in title and the number of words within the brands
def tag_brands(brand,title):
    tagging = ''
    brand = brand.split(' ')
    brand_started = False
    not_pass = False
    i = 0
    added_i = 0
    words = title.split(' ')
    for word in title.split(' '):
        if word == brand[0] and not_pass is False:
            tagging += 'B-B '
            brand_started = True
        elif len(brand) > 1 and brand_started:
            j = i
            for b in brand[1:]:
                #print(b,words[j],words,brand)
                if words[j] == b:
                    tagging += 'I-B '
                    added_i = added_i + 1
                else:
                    brand_started = False
                    tagging += 'O '
                    added_i = added_i + 1
                    
                j = j + 1
            brand_started = False
            not_pass = True
        else:
            brand_started = False
            if added_i >= 2:
                added_i = added_i - 1
            else:
                tagging += 'O '
                
        i = i + 1
        
    #tagging = ''
    #brand = brand.split(' ')
    #brand_started = False
    #for word in title.split(' '):
    #    if word == brand[0]:
    #        tagging += 'B-B '
    #        brand_started = True
    #    elif len(brand) > 1 and brand_started:
    #        for b in brand[1:]:
    #            if word == b:
    #                tagging += 'I-B '
    #            else:
    #                brand_started = False
    #                tagging += 'O '
    #    else:
    #        brand_started = False
    #        tagging += 'O '
    return tagging


def main(argv):
    if len(argv) >= 2:
        filename = sys.argv[1]
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            products = defaultdict(dict)
            line_count = 0
            suplemented = 0
            titles = []
            out = csv.writer(open("data/products.csv","w"))
            for row in csv_reader:
                if line_count > 0:
                    if(row[3]!='' and row[3]!=' '):
                        tmp_dict = {}
                        tmp_dict['SKUPARENT'] = row[1]
                        tmp_dict['MARCA']     = unescape(row[3]).lower()
                        tmp_dict['TITLE']     = unescape(row[9].split('|')[0].strip()).lower()
                        if not (tmp_dict['MARCA'] in tmp_dict['TITLE']):
                            suplemented += 1
                            title = tmp_dict['TITLE'] + ' ' + tmp_dict['MARCA']
                            tmp_dict['TITLE'] = title
                            titles.append(title)
                        tmp_dict['tag'] = tag_brands(tmp_dict['MARCA'],tmp_dict['TITLE'])
                        products[row[1]] = tmp_dict
                line_count = line_count + 1
            for k,v in products.items():
                out.writerow([v['TITLE'],v['MARCA'],v['tag']])



if __name__ == "__main__":
    main(sys.argv)

