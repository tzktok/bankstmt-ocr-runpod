import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
import postprocess
import numpy as np
import tempfile
from io import BytesIO
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from doctr.models import kie_predictor
from PIL import Image
from typing import List, Any
import cv2
import torch
import xml.etree.ElementTree as ET

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TableExtractor:
    def __init__(self):
        self.detection_model_path = 'bankstmt_ta .pt'
        self.structure_model_path = 'bankstmt_str.pt'
        self.device = 'cuda:0'
        self.detection_model = YOLO(self.detection_model_path).to(self.device)
        self.structure_model = YOLO(self.structure_model_path).to(self.device)
        self.detection_class_names = ['table', 'table rotated']
        self.ocr_model = PaddleOCR(use_gpu=True, lang='en', rec_model_dir="en_PP-OCRv4_rec_infer",
                                   det_model_dir="en_PP-OCRv3_det_infer")
        self.structure_class_names = ['table', 'table column', 'table column header',
                                      'table projected row header', 'table row', 'table spanning cell', 'no object']
        self.structure_class_map = {k: v for v, k in enumerate(self.structure_class_names)}
        self.structure_class_thresholds = {
            "table": 0.5,
            "table column": 0.5,
            "table column header": 0.5,
            "table projected row header": 0.5,
            "table row": 0.5,
            "table spanning cell": 0.5,
            "no object": 10
        }

    def table_detection(self, image):
        imgsz = 640
        pred = self.detection_model.predict(image, imgsz=imgsz)
        pred = pred[0].boxes
        result = pred.cpu().numpy()
        result_list = [list(result.xywhn[i]) + [result.conf[i], result.cls[i]] for i in range(result.shape[0])]
        return result_list

    def table_structure(self, image):
        imgsz = 800
        pred = self.structure_model.predict(image, imgsz=imgsz)
        pred = pred[0].boxes
        result = pred.cpu().numpy()
        result_list = [list(result.xywhn[i]) + [result.conf[i], result.cls[i]] for i in range(result.shape[0])]
        print(result_list)
        return result_list

    def crop_image(self, image, detection_result):

        crop_images = []
        width = image.shape[1]
        height = image.shape[0]
        for i, result in enumerate(detection_result):
            class_id = int(result[5])
            score = float(result[4])
            min_x = result[0]
            min_y = result[1]
            w = result[2]
            h = result[3]

            x1 = max(0, int((min_x - w / 2) * width) - 10)
            y1 = max(0, int((min_y - h / 2) * height) - 10)
            x2 = min(width, int((min_x + w / 2) * width) + 10)
            y2 = min(height, int((min_y + h / 2) * height) + 10)
            crop_image = image[y1:y2, x1:x2, :]
            crop_images.append(crop_image)

        return crop_images

    def get_word_bounding_boxes(self, line_box: List[List[float]], text: str) -> List[dict]:
        """
        Break down a line-level bounding box into word-level bounding boxes.

        :param line_box: List of 4 points representing the line bounding box
                         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        :param text: Recognized text for the line
        :return: List of dictionaries containing word-level information
        """

        def calculate_char_width(box: List[List[float]], text_length: int) -> float:
            """Estimate average character width based on bounding box and text length."""
            box_width = max(
                np.linalg.norm(np.array(box[1]) - np.array(box[0])),
                np.linalg.norm(np.array(box[2]) - np.array(box[3]))
            )
            return box_width / max(text_length, 1)  # Avoid division by zero

        def interpolate_points(start: List[float], end: List[float], ratio: float) -> List[int]:
            """Interpolate between two points."""
            return [int(start[i] + (end[i] - start[i]) * ratio) for i in range(2)]

        words = text.split()
        char_width = calculate_char_width(line_box, len(text))
        space_width = char_width * 0.7  # Assume space is 70% of char width

        result = []
        current_length = 0

        for word in words:
            word_length = len(word) * char_width
            word_start_ratio = current_length / (len(text) * char_width)
            word_end_ratio = (current_length + word_length) / (len(text) * char_width)

            # Calculate word bounding box
            word_box = [
                interpolate_points(line_box[0], line_box[1], word_start_ratio),
                interpolate_points(line_box[0], line_box[1], word_end_ratio),
                interpolate_points(line_box[3], line_box[2], word_end_ratio),
                interpolate_points(line_box[3], line_box[2], word_start_ratio)
            ]

            # Store in desired format
            formatted_bbox = [word_box[0][0], word_box[0][1], word_box[2][0], word_box[2][1]]
            result.append({'bbox': formatted_bbox, 'text': word})

            current_length += word_length + space_width

        return result

    def process_paddleocr_results(self, paddleocr_results: List[List[List[Any]]]) -> List[dict]:
        """
        Process PaddleOCR results to get word-level bounding boxes in the desired format.

        :param paddleocr_results: Results from PaddleOCR's ocr method
        :return: List of dictionaries with 'bbox' and 'text' for each word
        """
        word_level_results = []

        for page_result in paddleocr_results:
            if page_result is None:
                continue

            for line in page_result:
                if len(line) != 2:
                    continue
                line_box, (text, _) = line
                word_boxes = self.get_word_bounding_boxes(line_box, text)

                # Append each word box to the final result list
                word_level_results.extend(word_boxes)
        print(word_level_results)
        return word_level_results

    def ocr1(self, image):
        #ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = self.ocr_model.ocr(image, cls=True)
        final_word_level_results = self.process_paddleocr_results(result)
        return final_word_level_results

    def ocr(self, image):
        result = self.ocr_model.ocr(image, cls=True)
        result = result[0]
        new_result = []
        if result is not None:
            bounding_boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            for label, bbox in zip(txts, bounding_boxes):
                new_result.append({'bbox': [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]], 'text': label})
        print(new_result)
        return new_result

    def doctr(self, image):
        pil_image = Image.fromarray(image)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            # Save the image to the temporary file
            pil_image.save(temp_file, format='JPEG')
            temp_file_path = temp_file.name

        try:
            # Get image dimensions
            width, height = pil_image.size

            # Initialize the OCR model
            model = ocr_predictor(pretrained=True, det_arch="fast_base").to(device)

            # Load the document from the temporary file
            doc = DocumentFile.from_images(temp_file_path)

            # Perform OCR
            result = model(doc)

            # List to store formatted predictions
            formatted_predictions = []

            # Check if there are pages in the result
            if result.pages:
                # Iterate over pages
                for page in result.pages:
                    # Iterate over blocks in the page
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                # Access geometry
                                geometry = word.geometry
                                # Extract coordinates
                                x1, y1 = geometry[0]  # Top-left corner (x1, y1)
                                x2, y2 = geometry[1]  # Bottom-right corner (x2, y2)

                                # Create bbox in the desired format
                                bbox = [
                                    int(x1 * width),
                                    int(y1 * height),
                                    int(x2 * width),
                                    int(y2 * height)
                                ]

                                # Add word information to formatted predictions list
                                formatted_predictions.append({
                                    'bbox': bbox,
                                    'text': word.value,
                                })

            # Print or return formatted predictions
            print(formatted_predictions)
            return formatted_predictions

        finally:
            # Delete the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def convert_structure(self, words, image, structure_result):
        width = image.shape[1]
        height = image.shape[0]

        bboxes = []
        scores = []
        labels = []
        for i, result in enumerate(structure_result):
            class_id = int(result[5])
            score = float(result[4])
            min_x = result[0]
            min_y = result[1]
            w = result[2]
            h = result[3]

            x1 = int((min_x - w / 2) * width)
            y1 = int((min_y - h / 2) * height)
            x2 = int((min_x + w / 2) * width)
            y2 = int((min_y + h / 2) * height)

            bboxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(class_id)

        table_objects = []
        for bbox, score, label in zip(bboxes, scores, labels):
            table_objects.append({'bbox': bbox, 'score': score, 'label': label})

        table_class_objects = [obj for obj in table_objects if obj['label'] == self.structure_class_map['table']]
        if len(table_class_objects) > 1:
            table_class_objects = sorted(table_class_objects, key=lambda x: x['score'], reverse=True)
        try:
            table_bbox = list(table_class_objects[0]['bbox'])
        except:
            table_bbox = (0, 0, 1000, 1000)

        tokens_in_table = [token for token in words if postprocess.iob(token['bbox'], table_bbox) >= 0.5]

        table_structures, cells, confidence_score = postprocess.objects_to_cells(
            {'objects': table_objects, 'page_num': 0},
            table_objects, tokens_in_table,
            self.structure_class_names,
            self.structure_class_thresholds
        )
        print(table_structures)
        return table_structures, cells, confidence_score

    def visualize_cells(self, image, table_structures, cells):
        num_cols = len(table_structures['columns'])
        num_rows = len(table_structures['rows'])
        data_rows = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in cells:
            bbox = cell['bbox']
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            col_num = cell['column_nums'][0]
            row_num = cell['row_nums'][0]
            spans = cell['spans']
            text = ''
            for span in spans:
                if 'text' in span:
                    text += ' ' + span['text']
            data_rows[row_num][col_num] = text

            text_len = len(text)
            cell_width = x2 - x1
            num_per_line = cell_width // 10
            line_num = text_len // num_per_line if num_per_line != 0 else 0
            new_text = text[:num_per_line] + '\n'
            for j in range(line_num):
                new_text += text[(j + 1) * num_per_line:(j + 2) * num_per_line] + '\n'
        #print(data_rows)
        return data_rows

    def cells_to_html(self, cells):
        cells = sorted(cells, key=lambda k: min(k['column_nums']))
        cells = sorted(cells, key=lambda k: min(k['row_nums']))

        table = ET.Element("table")
        table.set('style', 'border-collapse: collapse;')
        current_row = -1

        for cell in cells:
            this_row = min(cell['row_nums'])

            attrib = {}
            colspan = len(cell['column_nums'])
            if colspan > 1:
                attrib['colspan'] = str(colspan)
            rowspan = len(cell['row_nums'])
            if rowspan > 1:
                attrib['rowspan'] = str(rowspan)
            if this_row > current_row:
                current_row = this_row
                if 'column header' in cell:
                    cell_tag = "th"
                    row = ET.SubElement(table, "thead")
                    row.set('style', 'border: 1px solid black;')
                else:
                    cell_tag = "td"
                    row = ET.SubElement(table, "tr")
                    row.set('style', 'border: 1px solid black;')
            tcell = ET.SubElement(row, cell_tag, attrib=attrib)
            tcell.set('style', 'border: 1px solid black; padding: 5px;')
            tcell.text = ''
            for span in cell['spans']:
                tcell.text += span['text'] + '\n'

        return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))


table_extractor = TableExtractor()
