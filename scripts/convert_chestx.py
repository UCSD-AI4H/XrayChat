import json
import os
import xml.etree.ElementTree as ET


def find(l, x):
    return [t for t in l if t.startswith(x)]


def main(img_path, report_path):
    img_files = os.listdir(img_path)
    out = []
    for file in os.listdir(report_path):
        ff = os.path.join(report_path, file)
        tree = ET.parse(ff)
        root = tree.getroot()
        # caption = root[16][0][2][3].text
        caption = ". ".join([root[16][0][2][ii].text.strip(". ") for ii in range(4) if root[16][0][2][ii].text is not None]) + "."

        assert caption, (caption, file)
        id0 = root[1].attrib['id'] + "_"
        img_f = find(img_files, id0)
        out.extend([{"image_id": i[:-4], "caption": caption} for i in img_f])

    print("total len", len(out))
    # return
    ob = {"annotations": out}

    with open("./chest_data/filter_cap.json", "wt") as f:
        json.dump(ob, f)


if __name__ == "__main__":
    img_path = "/data/youwei/llm/open-i/NLMCXR_png"
    report_path = "/data/youwei/llm/open-i/NLMCXR_reports/ecgen-radiology"
    main(img_path, report_path)
