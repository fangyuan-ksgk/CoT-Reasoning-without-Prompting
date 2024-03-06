import dataclasses
import json

@dataclasses.dataclass
class Attribute:
   name: str
   good_response: str
   bad_response: str
   scenario_desc: str
   weight: float = 1.0
   auto_compare_query: str = None # Use in auto-evaluaotor comparison
   anno_compare_query: str = None # Use in annotator comparison

@dataclasses.dataclass
class Requirement:
   attributes: list[Attribute] # this is a concise name for the attribute
   
   @classmethod
   def make(cls, data: list[dict[str, str]]):
       attributes = [Attribute(**item) for item in data]
       return cls(attributes=attributes)
   
   def add_scenario(self, scenario: dict[str, str]):
      self.attributes.append(Attribute(**scenario))

   def get_scenario_index(self, scenario_name: str):
      for idx in range(len(self.attributes)):
         if self.attributes[idx].name == scenario_name:
            return idx
      return -1

   def mutate_scenario(self, scenario: dict[str, str]) -> bool:
      index = self.get_scenario_index(scenario["name"])
      if index >= 0:
         self.attributes[index] = Attribute(**scenario)
         return True
      else:
         return False
      
   def get_anno_compare_queries(self):
      return [attribute.anno_compare_query for attribute in self.attributes]
   
   def get_auto_compare_queries(self):
      return [attribute.auto_compare_query for attribute in self.attributes]
   
   def get_attribute_names(self):
      return [attribute.name for attribute in self.attributes]
   
   def to_attribute_name_list(self):
      return [attribute.name for attribute in self.attributes]
   
   def to_attribute_dict(self):
      return [{"scenario_desc": a.scenario_desc, "good_response": a.good_response, "bad_response": a.bad_response} for a in self.attributes]
   
   @property
   def show_attributes(self):
      show_attributes = "\n".join([f"{attribute.name}\n- Scenario: {attribute.scenario_desc}\n- Good Response: {attribute.good_response}\n- Bad Response: {attribute.bad_response}\n" for attribute in self.attributes])
      return show_attributes
   
   def form_compare_template(self):
      
      template = """
      Please compare two conversations (conversation A, conversation B) and judge customer responses based on the following attributes:

      For each attribute described, identify whether the customer in Conversation A or Conversation B aligns more closely with the 'good response' or 'bad response'. 
      Your comparison response on each attributes should be a paragraph, and do not use new-line within the paragraph.

      {show_attributes}

      Based on these scenarios, evaluate and determine which customer (customer A or customer B) demonstrates more alignment with either 'good responses' or 'bad responses' in each attribute.

      Conversation A: {conversation_A}

      Conversation B: {conversation_B}
      """
      return template
   
   def form_compare_prompt(self, conversation_A, conversation_B):
      return self.form_compare_template().format(show_attributes=self.show_attributes, conversation_A=conversation_A, conversation_B=conversation_B)
   
   def save(self, filename: str):
      scenarios = [{"name": a.name, "scenario_desc": a.scenario_desc, "good_response": a.good_response, "bad_response": a.bad_response} for a in self.attributes]
      # store dict into json file
      with open(filename, 'w') as file:
          json.dump(scenarios, file, indent=4)
          
   @classmethod
   def load(cls, filename: str):
      # load dict from json file
      with open(filename, 'r') as file:
          scenarios = json.load(file)
      return cls.make(scenarios)
   
   def anno_compare_to_name(self, anno_compare_query: str):
      for attribute in self.attributes:
         if attribute.anno_compare_query == anno_compare_query:
            return attribute.name
      return None
   
   def auto_compare_to_name(self, auto_compare_query: str):
      for attribute in self.attributes:
         if attribute.auto_compare_query == auto_compare_query:
            return attribute.name
      return None
   
   def name_to_anno_compare(self, name: str):
      for attribute in self.attributes:
         if attribute.name == name:
            return attribute.anno_compare_query
      return None
   
   def name_to_auto_compare(self, name: str):
      for attribute in self.attributes:
         if attribute.name == name:
            return attribute.auto_compare_query
      return None
   
   def form_anno_compare_to_name_dict(self):
      return {attribute.anno_compare_query: attribute.name for attribute in self.attributes}
   
   def form_auto_compare_to_name_dict(self):
      return {attribute.auto_compare_query: attribute.name for attribute in self.attributes}
   
   def form_name_to_anno_compare_dict(self):
      return {attribute.name: attribute.anno_compare_query for attribute in self.attributes}
   
   def form_name_to_auto_compare_dict(self):
      return {attribute.name: attribute.auto_compare_query for attribute in self.attributes}
