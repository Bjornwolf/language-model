from xml.etree.ElementTree import iterparse
import os
import codecs
nos = ['1', '2', '3', '4']
ctr = 0
file_no = 1
for no in nos:
    print 'NO', no, ctr
    for event, elem in iterparse('plwiki-' + no + '.xml'):
        if elem.tag[-4:] == "page":
            title = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
            revision = elem.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
            content = revision.find('{http://www.mediawiki.org/xml/export-0.10/}text').text
            if title[:12] != "Wikipedysta:" and title[:21] != "Dyskusja wikipedysty:" and title[:11] != "Poczekalnia" and content != None:
                if ctr % 10000 == 0:
                    print ctr
                os.system("touch art" + str(ctr))
                out = codecs.open("art" + str(ctr), 'w', 'utf-8')
                out.write(content)
                out.close()
                ctr += 1

