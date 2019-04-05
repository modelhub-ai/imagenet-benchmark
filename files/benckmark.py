import requests
import json
import pandas as pd
import sys

# load imagenet synset
imagenet_synset = pd.read_csv('imagenet_synset.txt', sep=" ", header=None)
imagenet_synset.columns = ["synset", "imagenet_index", "imagenet_label"]
# load model synset
model_synset = pd.read_csv('model_synset_hash.txt', sep="#", header=None)
model_synset.columns = ["synset", "model_label"]
model_synset["model_index"] = range(1000)
# merge
df_synset = pd.merge(imagenet_synset, model_synset, on="synset")
df_synset.head()
# map
def imagenet_to_model_synset(imagenet_synset):
    return int(df_synset[df_synset["imagenet_index"]==imagenet_synset]["model_index"])

def generate_df(path_to_file):
    '''
    Generates and returns a pandas dataframe based on the ground thruth file from the Imagenet LSVRC.
    '''
    df = pd.read_csv(path_to_file, sep=" ", header=None)
    df.columns = ["imagenet_truth"]
    df["model_truth"] = [imagenet_to_model_synset(i) for i in list(df["imagenet_truth"])]
    df["pred"]=""
    df["CPU_time"]=""
    df["top_1"]=0
    df["top_5"]=0
    df.head()
    return df

def parse_prediction(dictionary):
    '''
    Returns the top 5 indices.
    '''
#    d_sorted_label = sorted(dictionary, key=lambda x: x["probability"], reverse=True)
    d_sorted_index = sorted(range(1000), key=lambda i: dictionary[i]["probability"], reverse=True)
    top5 = d_sorted_index[:5]
    return top5

def populate_df(file_list, df, model_name, method="POST", port=80):
    '''
    file_list could either be a list of urls pointing to images in which case method should be
    set to GET, or a list of file names in which case method should be POST.
    '''

    for index, item in enumerate(file_list):

        # REST API
        if method == "GET":
            r = requests.get('http://localhost:' + str(port) + '/api/predict?fileurl=' + item)
        elif method == "POST":
            files = {'file': open('/data/ILSVRC2012_img_val/' + item, 'r')}
            r = requests.post('http://localhost:' + str(port) + '/api/predict', files=files)

        dictionary = json.loads(r.text)
        predicitons = dictionary["output"][0]["prediction"]

        # parse predictions and insert into dataframe
        top5_pred = parse_prediction(predicitons)
        df.at[index, 'pred'] = top5_pred

        # check top_1 accuracy
        if df.iloc[index]['model_truth'] == top5_pred[0]:
            df.at[index, 'top_1'] = 1
        # check top_5 accuracy
        if df.iloc[index]['model_truth'] in top5_pred:
            df.at[index, 'top_5'] = 1

        # time
        df.at[index, 'CPU_time'] = dictionary["processing_time"]

        # save df
        df.to_pickle("/output/" + model_name)

        print "image at index {} done".format(index)

if __name__ == "__main__":
    try:
        df = generate_df('/data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
        # file_list = ["bagel.jpg", "bottle.jpeg", "cat.jpeg", "laptop.jpeg", "pizza.jpeg"]
        file_list = [ "ILSVRC2012_val_" + str(i).zfill(8) + ".JPEG" for i in range(1,50001) ]
        populate_df(file_list, df, sys.argv[1])
    except Exception as e:
        print (e)
