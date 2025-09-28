#!/usr/bin/env python3
"""
Test script to demonstrate chunk preview logging
This shows how the new chunk preview functionality works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from minirag.operate import _get_chunk_preview

def test_chunk_preview():
    """Test the chunk preview function with various content lengths"""
    
    print("🧪 Testing Chunk Preview Function")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "Short text",
        "This is a medium length text that should be handled properly by the preview function and show the full content.",
        "This is a very long text that contains many words and should be truncated to show only the first few words and the last few words with an ellipsis in between to demonstrate the chunk preview functionality that we just implemented in the MiniRAG system.",
        "Kalpana Chawla was an Indian-American astronaut and the first woman of Indian origin to go to space. She was one of seven crew members who died in the Space Shuttle Columbia disaster on February 1, 2003. Chawla was posthumously awarded the Congressional Space Medal of Honor, and several streets, universities, and institutions have been named in her honor. She was born in Karnal, Haryana, India, and completed her early education there before moving to the United States for higher studies. Chawla earned a Bachelor of Science degree in aeronautical engineering from Punjab Engineering College in 1982, and then moved to the United States to pursue a Master of Science degree in aerospace engineering from the University of Texas at Arlington in 1984. She later earned a Doctor of Philosophy degree in aerospace engineering from the University of Colorado Boulder in 1988. After completing her education, Chawla worked as a research scientist at NASA Ames Research Center and later joined the NASA Astronaut Corps in 1995. She flew on two space missions: STS-87 in 1997 and STS-107 in 2003, the latter being her final mission that ended in tragedy.",
        "The Federal Reserve System is the central banking system of the United States. It was created on December 23, 1913, with the enactment of the Federal Reserve Act, after a series of financial panics led to the desire for central control of the monetary system in order to alleviate financial crises. Over the years, events such as the Great Depression in the 1930s and the Great Recession during the 2000s have led to the expansion of the roles and responsibilities of the Federal Reserve System. The Federal Reserve System's structure is composed of the presidentially appointed Board of Governors, the Federal Open Market Committee, twelve regional Federal Reserve Banks located in major cities throughout the nation, numerous privately owned U.S. member banks and various advisory councils. The Federal Reserve System's duties have expanded over the years, and today, according to official Federal Reserve documentation, include conducting the nation's monetary policy, supervising and regulating banking institutions, maintaining the stability of the financial system and providing financial services to depository institutions, the U.S. government, and foreign official institutions. The Federal Reserve System is considered an independent central bank because its monetary policy decisions do not have to be approved by the President or anyone else in the executive or legislative branches of government, it does not receive funding appropriated by Congress, and the terms of the members of the Board of Governors span multiple presidential and congressional terms."
    ]
    
    for i, content in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original length: {len(content)} characters")
        print(f"Word count: {len(content.split())} words")
        print(f"Preview: {_get_chunk_preview(content)}")
        print("-" * 30)

if __name__ == "__main__":
    test_chunk_preview()
