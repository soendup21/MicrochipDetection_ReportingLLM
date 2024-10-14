วิธีใช้

////*ไฟล์โค้ด.py โฟลเดอร์รูป(.jpg & .xml) โฟลเดอร์รูปที่ต้องการให้ออก ต้องอยู่ในโฟลเดอร์เดียวกัน////
main folder
	>code.py
	>folder of picture you gonna label
		>.jpg
		>.xml(will auto create after label in labelImg)
	>folder of picture of labeled image

1.เข้า VS code
2.พิมพ์ labelImg ใน terminal
3.เปิด folder ที่ต้องการ label ภาพ (ภาพต้องเป็น .jpg) 
4.ครอบภาพที่ต้องการ หลังเราครอบหรือทำlabel จะมีไฟล์ .xml
5.เข้าโค้ดแล้วเปลี่ยน directory โฟลเดอร์
# Example usage
image_dir = r'convert_data_type_code\Trays_and_annotate_all'  # Directory where your images are
xml_dir = r'convert_data_type_code\Trays_and_annotate_all'  # Directory where your XML files are
output_dir = r'convert_data_type_code\jpg_image'  # Directory where cropped images will be saved