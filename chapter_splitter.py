# 章节分割器（学习资料专用）
import re
import logging
from typing import List, Tuple, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChapterSplitter:
    """
    按章节分割文本，以小标题作为最小分割单位
    """
    
    def __init__(self, heading_patterns: List[str], overlap_lines: int = 2, min_content_length: int = 50):
        """
        初始化章节分割器
        
        Args:
            heading_patterns: 标题匹配正则表达式列表
            overlap_lines: 章节之间的重叠行数
            min_content_length: 章节最小内容长度（字符）
        """
        self.heading_patterns = heading_patterns
        self.overlap_lines = overlap_lines
        self.min_content_length = min_content_length
        logger.info(f"章节分割器初始化: 标题模式={len(heading_patterns)}个, 重叠行数={overlap_lines}, 最小长度={min_content_length}")
    
    def is_heading(self, line: str) -> bool:
        """
        判断一行是否是标题
        
        Args:
            line: 待判断的文本行
        
        Returns:
            True 如果是标题，False 否则
        """
        line = line.strip()
        if not line:
            return False
        
        for pattern in self.heading_patterns:
            if re.match(pattern, line):
                logger.debug(f"匹配标题模式 '{pattern}': {line}")
                return True
        return False
    
    def split_by_chapters(self, text: str) -> List[Tuple[str, str]]:
        """
        按章节分割文本
        
        Args:
            text: 原始文本
        
        Returns:
            章节列表，每个元素是 (章节标题, 章节内容)
        """
        if not text or not text.strip():
            logger.info("文本为空，返回空列表")
            return []
        
        # 将文本按行分割
        lines = text.split('\n')
        
        chapters = []
        current_chapter_title = None
        current_chapter_lines = []
        
        # 第一遍扫描：识别章节标题并初步分割
        for i, line in enumerate(lines):
            # 如果当前行是标题
            if self.is_heading(line):
                # 如果已有当前章节，先保存
                if current_chapter_title and current_chapter_lines:
                    chapter_content = '\n'.join(current_chapter_lines).strip()
                    if len(chapter_content) >= self.min_content_length:
                        chapters.append((current_chapter_title, chapter_content))
                        logger.debug(f"保存章节: {current_chapter_title}, 内容长度: {len(chapter_content)}")
                    else:
                        logger.debug(f"跳过短章节: {current_chapter_title}, 内容长度: {len(chapter_content)}")
                
                # 开始新章节
                current_chapter_title = line.strip()
                current_chapter_lines = []
            else:
                # 添加到当前章节
                current_chapter_lines.append(line)
        
        # 保存最后一个章节
        if current_chapter_title and current_chapter_lines:
            chapter_content = '\n'.join(current_chapter_lines).strip()
            if len(chapter_content) >= self.min_content_length:
                chapters.append((current_chapter_title, chapter_content))
                logger.debug(f"保存最后章节: {current_chapter_title}, 内容长度: {len(chapter_content)}")
        
        logger.info(f"初步分割完成，共识别 {len(chapters)} 个章节")
        
        # 如果没有识别到任何章节，返回原始文本作为一个章节
        if not chapters:
            logger.info("未识别到章节标题，将整篇作为一个章节")
            return [("全文", text.strip())]
        
        # 添加重叠内容
        chapters_with_overlap = []
        for i, (title, content) in enumerate(chapters):
            # 获取前一章的末尾内容作为重叠
            if i > 0 and self.overlap_lines > 0:
                prev_content = chapters[i-1][1]
                prev_lines = prev_content.split('\n')[-self.overlap_lines:]
                overlap_content = '\n'.join(prev_lines)
                content = overlap_content + '\n' + content
            
            chapters_with_overlap.append((title, content))
        
        logger.info(f"添加重叠后，共 {len(chapters_with_overlap)} 个章节")
        
        return chapters_with_overlap
    
    def split_with_metadata(self, text: str, filename: str) -> List[Dict]:
        """
        按章节分割文本并生成元数据
        
        Args:
            text: 原始文本
            filename: 文件名
        
        Returns:
            文档块列表，每个文档块包含 content 和 metadata
        """
        chapters = self.split_by_chapters(text)
        
        docs_with_metadata = []
        for i, (chapter_title, chapter_content) in enumerate(chapters):
            metadata = {
                'source': filename,
                'chapter': chapter_title,
                'chapter_number': i + 1,
                'total_chapters': len(chapters),
                'content_length': len(chapter_content),
            }
            docs_with_metadata.append({
                'content': chapter_content,
                'metadata': metadata
            })
            
            logger.debug(f"生成文档块: 章节 {i+1}/{len(chapters)} - {chapter_title}")
        
        return docs_with_metadata


# 测试代码
if __name__ == "__main__":
    # 测试文本
    test_text = """第一章 心理学导论

心理学是研究心理现象及其规律的科学。它涉及认知、情感、行为等多个方面。

1.1 什么是心理学

心理学不仅研究人类的心理活动，也研究动物的行为。

一、心理学的研究对象

心理学的研究对象包括心理过程和个性心理两个方面。

第二章 认知心理学

认知心理学研究人类的认知过程，包括感知、记忆、思维等。

2.1 感知觉

感知觉是人类认识世界的基础。
"""
    
    # 创建分割器
    patterns = [
        r'^第[零一二三四五六七八九十百千万]+[章节编篇部卷].*$',
        r'^\d+[\.\uff0e、][^\n]+$',
        r'^[一二三四五六七八九十]+[\uff0e、\.][^\n]+$',
    ]
    
    splitter = ChapterSplitter(patterns, overlap_lines=1)
    docs = splitter.split_with_metadata(test_text, "心理学导论.txt")
    
    print("分割结果:")
    for doc in docs:
        print(f"\n=== 章节 {doc['metadata']['chapter_number']}: {doc['metadata']['chapter']} ===")
        print(f"来源: {doc['metadata']['source']}")
        print(f"内容长度: {doc['metadata']['content_length']}")
        print(f"内容预览:\n{doc['content'][:200]}...")
