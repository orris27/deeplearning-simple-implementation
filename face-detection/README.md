## Common APIs
### Face Recognition
1. Smiling: [https://www.faceplusplus.com.cn/face-detection/#demo]
```
curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key=<api_key>"  -F "api_secret=<api_secret>"  -F "image_url=https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1564113226&di=57d37113aca7d923605b5a30944ebc93&imgtype=jpg&er=1&src=http%3A%2F%2Fww2.sinaimg.cn%2Forj480%2Fa721d309gw1f8ui6bg1f9j20n50yqgpq.jpg" -F "return_landmark=0"  -F "return_attributes=smiling" | python -mjson.tool

# Another choice is to send binary data with POST multipart/form-data
#curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key=<api_key>"  -F "api_secret=<api_secret>"  -F "image_file=@/tmp/1.jpeg" -F "return_landmark=0"  -F "return_attributes=smiling" | python -mjson.tool


# Outputs
{
    "time_used": 271,
    "faces": [
        {
            "attributes": {
                "smile": {
                    "threshold": 50.0,
                    "value": 100.0
                }
            },
            "face_rectangle": {
                "width": 214,
                "top": 204,
                "left": 130,
                "height": 214
            },
            "face_token": "706e4382091cf843d4c651e214f72361"
        }
    ],
    "image_id": "J6SxH10JZSNGBfAoTiijdw==",
    "request_id": "1563508595,a168b2bb-bcdc-478a-9590-f240a1b24bdb",
    "face_num": 1
}
```
2. 1:1. see details [here](https://github.com/ageitgey/face_recognition)
```
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```
